"""
Barebones model-based design of experiments (mbDOE) with FIM / D-optimality.

Frequentist counterpart to the HSA_K series. Same LHHW model (HSA_Mo_B_011),
same design space (HSA_K_001), but no priors and no sampling:

    1. fit theta by weighted least squares                 (jax BFGS)
    2. FIM(theta_hat) = sum_i J_i^T J_i / sigma_i^2        (jax autodiff)
    3. x_next = argmax_x logdet(FIM_D + FIM(x))            (jax BFGS, multi-start)
    4. run the experiment (synthetic), append, repeat

Starts from 8 experiments (--init-design):
  factorial  2^(4-1) fractional factorial over T, p, c2h, c2w (c2w = T*p*c2h),
             c2m=1000, c2i=0.071. Full-rank FIM, all 8 parameters identifiable.
  grid       fixed composition (c2h=0.2, c2m=c2w=1000, c2i=0.071), 3x3 grid over
             T x p without the center point. Only 5 of 8 parameters identifiable.

Every step is logged to {datadir}/{EXP_ID}/seed{seed:03d}/: step_XXX.json holds
the design so far, observations, fitted parameters, FIM and the next proposal;
experiment_summary.json collects all steps; run.log mirrors the console.

Noise model matches generate_synthetic_data: sigma_i = sigma_rel * |r_i|, so the
per-point FIM contribution is grad(log r_i)^T grad(log r_i) / sigma_rel^2.

Parameters are estimated in a transformed space (log for positive quantities,
dH / 1e4 for enthalpies) so the FIM is well scaled and det() is meaningful.
"""

import argparse
import itertools
import datetime
import json
import logging
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import namer
from jax.scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- model (from HSA_Mo_B_011)
# Copied rather than imported: the deterministic part of HSA_Mo_B_011 without
# numpyro (numpyro 0.20 does not import with the locked jax 0.11.2).
TRUE_PARAMS = {
    "k0": 3.46e-4,
    "EA": 77.5e3,
    "A_OH": 0.50,
    "H_OH": 22.4e3,
    "A_H2": 0.44,
    "H_H2": -6.2e3,
    "A_mix": 0.88,
    "H_mix": -10.0e3,
}


def kinetics_model(p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix):
    """LHHW rate for CO2 + 4H2 <-> CH4 + 2H2O. p_i: (n, 5), T: (n, 1) -> (n, 1)."""
    R_const, T_ref = 8.314, 555.0
    p_i = jnp.reshape(p_i, (-1, 5))
    T = jnp.reshape(T, (-1, 1))
    arr = lambda E: jnp.exp(E / R_const * (1.0 / T_ref - 1.0 / T))

    k = k0 * arr(Ea_k)
    K_OH, K_H2, K_mix = A_oh * arr(dH_oh), A_h2 * arr(dH_h2), A_mix * arr(dH_mix)
    K_eq = 137.0 * T ** (-4.0) * jnp.exp(158700.0 / (R_const * T))

    p_CO2, p_H2, p_H2O, p_CH4 = (p_i[:, j : j + 1] for j in range(4))
    driving_force = 1.0 - (p_CH4 * p_H2O**2.0) / (p_CO2 * p_H2**4.0 * K_eq)
    inhibition = (
        1.0 + K_OH * p_H2O / jnp.sqrt(p_H2) + K_H2 * jnp.sqrt(p_H2) + K_mix * jnp.sqrt(p_CO2)
    ) ** 2.0
    return k * jnp.sqrt(p_H2) * jnp.sqrt(p_CO2) * driving_force / inhibition


def calculate_partial_pressures(p_total, c2h, c2m, c2w, c2i):
    """Ratios (n, 1) -> partial pressures (n, 5) ordered [CO2, H2, H2O, CH4, I]."""
    x_co2 = (1 / (1 + 1 / c2h + 1 / c2m + 1 / c2w + 1 / c2i)).squeeze(-1)
    xi = jnp.stack(
        (x_co2, x_co2 / c2h.squeeze(-1), x_co2 / c2w.squeeze(-1),
         x_co2 / c2m.squeeze(-1), x_co2 / c2i.squeeze(-1)),
        axis=1,
    )
    return p_total * xi


def generate_synthetic_data(T, p, c2h, c2m, c2w, c2i, rel_noise_std=0.05, seed=0):
    key = jax.random.PRNGKey(seed)
    p_i = calculate_partial_pressures(p, c2h, c2m, c2w, c2i)
    r_true = kinetics_model(p_i, T, *TRUE_PARAMS.values())
    r_obs = r_true + jax.random.normal(key, r_true.shape) * rel_noise_std * jnp.abs(r_true)
    return T, p_i, r_obs


PARAM_NAMES = ("k0", "Ea_k", "A_oh", "dH_oh", "A_h2", "dH_h2", "A_mix", "dH_mix")
LOG_PARAMS = jnp.array([1, 1, 1, 0, 1, 0, 1, 0], dtype=bool)
DH_SCALE = 1e4

# Design space from HSA_K_001: (lower, upper, log-scaled)
DESIGN_NAMES = ("T", "p", "c2h", "c2m", "c2w", "c2i")
X_LB = jnp.array([450.0, 3.0, 0.2, 0.5, 0.5, 0.05])
X_UB = jnp.array([600.0, 9.0, 4.0, 1000.0, 1000.0, 1.0])
X_LOG = jnp.array([0, 0, 1, 1, 1, 1], dtype=bool)

REL_NOISE = 0.1  # same as HSA_K_001

# Initial design "grid": low/mid/high T and p at fixed composition, center point dropped
INIT_COMPOSITION = (0.2, 1000.0, 1000.0, 0.071)  # c2h, c2m, c2w, c2i
INIT_TP = [(T, p) for T in (450.0, 525.0, 600.0) for p in (3.0, 6.0, 9.0)
           if (T, p) != (525.0, 6.0)]
INIT_GRID = [[T, p, *INIT_COMPOSITION] for T, p in INIT_TP]

# Initial design "factorial": 2^(4-1), generator c2w = T*p*c2h (resolution IV).
# Water has to vary, otherwise A_oh / dH_oh are not identifiable.
INIT_FACTORIAL = [
    [(450.0, 600.0)[T], (3.0, 9.0)[p], (0.2, 4.0)[h], 1000.0, (1000.0, 0.5)[(T + p + h) % 2], 0.071]
    for T, p, h in itertools.product((0, 1), repeat=3)
]
INIT_DESIGNS = {"factorial": INIT_FACTORIAL, "grid": INIT_GRID}

# Tiny ridge so logdet stays finite while the FIM is (near) rank deficient
RIDGE = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=32, help="Sequential D-opt experiments")
parser.add_argument("--n-starts", type=int, default=16, help="Multi-starts for the design optimizer")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--init-design", choices=list(INIT_DESIGNS), default="factorial")
parser.add_argument("--datadir", type=Path, default=Path.cwd() / "data")


# ---------------------------------------------------------------- transforms
def to_phys(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(LOG_PARAMS, jnp.exp(theta), theta * DH_SCALE)


def to_theta(phys: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(LOG_PARAMS, jnp.log(jnp.abs(phys)), phys / DH_SCALE)


def u_to_x(u: jnp.ndarray) -> jnp.ndarray:
    """Unconstrained u -> design x inside the box (log-uniform where X_LOG)."""
    s = jax.nn.sigmoid(u)
    lin = X_LB + s * (X_UB - X_LB)
    log = jnp.exp(jnp.log(X_LB) + s * (jnp.log(X_UB) - jnp.log(X_LB)))
    return jnp.where(X_LOG, log, lin)


def rate(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Reaction rate for a single design x (6,) -> scalar."""
    p_i = calculate_partial_pressures(
        x[1], x[2:3, None], x[3:4, None], x[4:5, None], x[5:6, None]
    )
    return kinetics_model(p_i, x[0], *to_phys(theta)).squeeze()


rates = jax.vmap(rate, in_axes=(None, 0))


# ---------------------------------------------------------------- estimation
def fit(theta0: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Weighted least squares with relative residuals (matches the noise model)."""

    def loss(theta):
        return jnp.sum(((y - rates(theta, X)) / jnp.abs(y)) ** 2)

    res = minimize(loss, theta0, method="BFGS", options={"maxiter": 2000})
    return res.x


# ---------------------------------------------------------------- FIM
def fim(theta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """FIM = S^T S / sigma_rel^2 with S_ij = d log r_i / d theta_j."""
    S = jax.jacfwd(lambda t: jnp.log(jnp.abs(rates(t, X))))(theta)
    return S.T @ S / REL_NOISE**2


def logdet(M: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.slogdet(M + RIDGE * jnp.eye(M.shape[0]))[1]


# ---------------------------------------------------------------- design
def next_experiment(theta: jnp.ndarray, X: jnp.ndarray, key, n_starts: int):
    """argmax_x logdet(FIM(X) + FIM(x)) via multi-start BFGS."""
    F0 = fim(theta, X)

    def neg_logdet(u):
        return -logdet(F0 + fim(theta, u_to_x(u)[None, :]))

    solve = jax.jit(jax.vmap(lambda u0: minimize(neg_logdet, u0, method="BFGS").x))
    U = solve(jax.random.normal(key, (n_starts, X.shape[1])) * 1.5)
    vals = jax.vmap(neg_logdet)(U)
    vals = jnp.where(jnp.isfinite(vals), vals, jnp.inf)
    best = jnp.argmin(vals)
    return u_to_x(U[best]), -vals[best]


def experiment(X: jnp.ndarray, seed: int) -> jnp.ndarray:
    """Synthetic measurement at designs X, same generator as HSA_K_001."""
    _, _, r_obs = generate_synthetic_data(
        X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4], X[:, 4:5], X[:, 5:6],
        rel_noise_std=REL_NOISE, seed=seed,
    )
    return r_obs.squeeze(-1)


def report(i: int, theta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, save_path: Path, x_next=None, ld_next=None) -> dict:
    """Log the current step and write it to step_XXX.json."""
    F = fim(theta, X)
    # ridge (not pinv): unidentifiable directions show up as huge sd instead of zero
    sd = jnp.sqrt(jnp.diag(jnp.linalg.inv(F + RIDGE * jnp.eye(F.shape[0]))))
    phys = to_phys(theta)
    true = list(TRUE_PARAMS.values())
    logger.info(f"\n[{i:3d}] n={X.shape[0]:3d}  logdet(FIM)={logdet(F):9.3f}  rank={jnp.linalg.matrix_rank(F)}")
    logger.info(f"      {'param':>7} {'estimate':>11} {'true':>11} {'sd(theta)':>10}")
    for n, e, t, s in zip(PARAM_NAMES, phys, true, sd):
        logger.info(f"      {n:>7} {e:11.4g} {t:11.4g} {s:10.3g}")
    if x_next is not None:
        logger.info("      next: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(DESIGN_NAMES, x_next)))
        logger.info(f"      expected logdet(FIM) after next: {ld_next:.3f}")

    step = {
        "step": i,
        "n_experiments": int(X.shape[0]),
        "observed": {"x": X.tolist(), "y": y.tolist(), "x_names": list(DESIGN_NAMES)},
        "parameters": {n: float(v) for n, v in zip(PARAM_NAMES, phys)},
        "theta": theta.tolist(),
        "sd_theta": sd.tolist(),
        "fim": F.tolist(),
        "logdet_fim": float(logdet(F)),
        "next_proposal": None if x_next is None else {
            "x": dict(zip(DESIGN_NAMES, map(float, x_next))),
            "expected_logdet_fim": float(ld_next),
        },
    }
    with open(save_path / f"step_{i:03d}.json", "w") as f:
        json.dump(step, f, indent=2)
    return step


def main() -> None:
    args = parser.parse_args()
    exp_id = namer.generate()
    save_path = args.datadir / exp_id / f"seed{args.seed:03d}"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(save_path / "run.log"))

    logger.info("=" * 70)
    logger.info(f"FIM D-OPTIMAL mbDOE  |  Experiment ID: {exp_id}  |  Seed: {args.seed}")
    logger.info(f"  Iterations: {args.iterations}, multi-starts: {args.n_starts}, initial design: {args.init_design}")
    logger.info(f"  Results: {save_path}")
    logger.info("=" * 70)

    key = jax.random.PRNGKey(args.seed)
    X = jnp.array(INIT_DESIGNS[args.init_design])
    y = experiment(X, seed=args.seed)

    # starting guess: prior means of HSA_Mo_B_011.bayesian_model
    theta = to_theta(jnp.array([1e-3, 6e4, 1.0, 1e4, 1.0, -1e4, 1.0, -1e4]))
    steps = []

    for i in range(args.iterations + 1):
        theta = fit(theta, X, y)  # warm start from previous estimate
        if i == args.iterations:
            steps.append(report(i, theta, X, y, save_path))
            break
        key, sub = jax.random.split(key)
        x_new, ld = next_experiment(theta, X, sub, args.n_starts)
        steps.append(report(i, theta, X, y, save_path, x_new, ld))

        X = jnp.vstack([X, x_new[None, :]])
        y = jnp.concatenate([y, experiment(x_new[None, :], seed=args.seed + i + 1)])

    summary = {
        "experiment_id": exp_id,
        "seed": args.seed,
        "timestamp": str(datetime.datetime.now()),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "true_parameters": dict(zip(PARAM_NAMES, TRUE_PARAMS.values())),
        "steps": steps,
    }
    with open(save_path / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nDone. Experiment ID: {exp_id}, results in {save_path}")


if __name__ == "__main__":
    main()
