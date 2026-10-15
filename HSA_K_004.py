"""
Bayesian Optimal Experimental Design using an inference-free PREDICTIVE EIG (NumPyro)

Companion to HSA_K_001, which scores a candidate by what it teaches about the
PARAMETERS. This script scores a candidate by what it teaches about the MODEL
PREDICTIONS on a fixed test grid:

    U(x) = E_y[ KL( p(ytilde | y, x, xtest, D) || p(ytilde | xtest, D) ) ]
         = I(ytilde ; y | x, xtest, D)

where ytilde is the JOINT K-vector of predictions on the grid, not K separate
marginals. The old HSA_H_003 summed a per-test-point KLD, which counts the same
information once per test point -- all K predictions are driven by the same nine
parameters, so they are massively redundant. That overcounts, the opposite
direction to the parametric criterion's undercounting.

As in K_001 no refit is needed. The enabling structure is that y and ytilde are
conditionally independent given theta -- both are the rate law plus noise at
different conditions -- so

    p(y, ytilde) = E_theta[ p(y|theta) p(ytilde|theta) ]

and all three densities in

    I = E[ log p(y, ytilde) - log p(y) - log p(ytilde) ]

are mixture averages over draws from the current posterior.

Two practical consequences:
- The test-grid term does not depend on the candidate, so it is computed ONCE per
  design iteration and reused across the whole Ax search. A candidate then costs
  one forward call and one (N, M) matrix, the same as the parametric criterion.
- There is no self-term, so the PCE lower bound of K_001 does not transfer.
  Convergence in M is checked by halving the contrastive set instead.

Validity check, logged every iteration: ytilde is a noisy function of theta, so by
the data processing inequality I(ytilde; y) <= I(theta; y). A predictive value above
the parametric one is provably an artifact, not a judgement call.

Each seed run:
- Uses the same EXP_ID
- Stores results in {datadir}/{EXP_ID}/seed{seed:03d}/
- Generates synthetic data with the specified seed

Key Features:
- Inference-free nested-Monte-Carlo estimate of the joint predictive EIG
- Test-grid term precomputed once per design iteration
- Common random numbers across candidates within a design iteration
- Data-processing-inequality check against the parametric EIG
- Integration with Ax platform for Bayesian optimization
"""

import gc
import os
import json
import argparse
from pathlib import Path
import datetime
import logging
import time
import arviz as az
import numpy as np

from scipy.special import logsumexp
from scipy.stats import qmc
import xarray
import namer
from ax import Client, RangeParameterConfig
import numpyro

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Host devices numpyro may use for parallel chains. Defaults to the cluster
# allocation; export BIBO_HOST_DEVICES=1 to force fully sequential chains.
# numpyro.set_host_device_count(int(os.environ.get("BIBO_HOST_DEVICES", "48")))
numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)
import jax
import jax.numpy as jnp


from HSA_Mo_B_012 import (
    run_inference,
    sample_prior,
    generate_synthetic_data,
    calculate_partial_pressures,
    kinetics_model,
)

# Parameter vector layout. The first eight are the kinetic parameters consumed by
# kinetics_model (in its positional order); the ninth is the relative noise scale.
PARAM_ORDER = (
    "k0",
    "Ea_k",
    "A_oh",
    "dH_oh",
    "A_h2",
    "dH_h2",
    "A_mix",
    "dH_mix",
    "sigma_r",
)

# Matches the likelihood in the model module's bayesian_model:
#     sigma_obs = sigma_r * |r_pred| + 1e-10
# The additive term is load-bearing: without it sigma reaches ~1e-15 on early
# posteriors and the log-densities span 1e21.
SIGMA_FLOOR = 1e-10

# Single source of truth for the design box: the Ax search space and the test grid
# must live in the same box, or the criterion is scored outside the region searched.
# Third entry is the Ax scaling ("log" for the ratio parameters, which span orders
# of magnitude and otherwise crush all the variation into the first 1% of the axis).
DESIGN_SPEC = (
    ("T", (450.0, 600.0), None),
    ("p", (3.0, 9.0), None),
    ("c2h", (0.2, 4.0), "log"),
    ("c2m", (0.5, 1000.0), "log"),
    ("c2w", (0.5, 1000.0), "log"),
    ("c2i", (0.05, 1.0), "log"),
)
DESIGN_BOUNDS = [bounds for _, bounds, _ in DESIGN_SPEC]

# The test grid is fixed for the whole study so the objective means the same thing
# across design iterations, seeds and campaigns.
TEST_GRID_SEED = 0

# An outer sample counts as saturated when its contribution sits within this many
# nats of the log(M+1) ceiling, i.e. the contrastive sum has collapsed onto the
# self-term and the value reflects n_inner rather than the information content.
SATURATION_TOL = 0.1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-outer",
    type=int,
    default=512,
    help="Outer Monte Carlo draws N for the EIG estimator",
)
parser.add_argument(
    "--n-inner",
    type=int,
    default=4096,
    help="Contrastive draws M for the mixture densities in the EIG estimator",
)
parser.add_argument(
    "--ntestpoints",
    type=int,
    default=20,
    help="Number of test-grid points the predictive EIG is evaluated against",
)
parser.add_argument(
    "--prior-samples",
    type=int,
    default=16384,
    help="Prior draws for the iteration-0 design (must be >= n-outer + n-inner)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=32,
    help="Number of points to additionally sample in the experiment",
)
parser.add_argument(
    "--optiterations",
    type=int,
    default=30,
    help="Number of optimization iterations for Ax",
)
parser.add_argument(
    "--trialsperiteration",
    type=int,
    default=4,
    help="Number of trials per optimization iteration for Ax",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Base random seed for reproducible workflow execution",
)
parser.add_argument(
    "--n-seeds",
    type=int,
    default=1,
    help="Number of different seeds to run (for ensemble studies)",
)
parser.add_argument(
    "--datadir",
    type=Path,
    default=Path.cwd() / "data",
    help="Base directory for run artifacts",
)


def draw_theta(
    idata: xarray.DataTree,
    n_draws: int,
    key: jax.Array,
    group: str = "posterior",
) -> np.ndarray:
    """
    Draw parameter vectors from a posterior (or prior) group.

    Stacks chain/draw and takes a key-derived permutation. Unlike
    ``az.extract(num_samples=...)`` this is seeded, so the acquisition is
    reproducible.

    Args:
        idata: InferenceData / DataTree holding the group
        n_draws: Number of draws to return
        key: JAX PRNG key selecting the subsample
        group: Group to read from

    Returns:
        (n_draws, 9) float64 array, columns ordered as PARAM_ORDER
    """
    stacked = az.extract(
        idata,
        group=group,
        var_names=list(PARAM_ORDER),
        num_samples=None,
    )
    theta = np.stack(
        [np.asarray(stacked[v].values, dtype=np.float64).ravel() for v in PARAM_ORDER],
        axis=1,
    )
    n_available = theta.shape[0]
    if n_available < n_draws:
        raise ValueError(
            f"Requested {n_draws} draws but group '{group}' holds only {n_available}. "
            f"Raise --prior-samples (iteration 0) or the run_inference draw count "
            f"(later iterations), or lower --n-outer / --n-inner."
        )
    idx = np.asarray(jax.random.permutation(key, n_available)[:n_draws])
    return theta[idx]


def rates_at(x_new: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Forward-model the reaction rate at one design point for many parameter vectors.

    Thin wrapper over rates_grid for the single-candidate case.

    Args:
        x_new: Design point (6,) [T, p, c2h, c2m, c2w, c2i]
        theta: (K, 9) parameter draws, columns ordered as PARAM_ORDER

    Returns:
        (K,) float64 array of predicted rates
    """
    return rates_grid(np.asarray(x_new, dtype=np.float64).reshape(1, 6), theta)[0]


def rates_grid(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Forward-model the reaction rate at several design points for many draws.

    One call covers the whole grid and every draw, since kinetics_model is already
    vectorised over both axes.

    Args:
        X: (n, 6) design rows [T, p, c2h, c2m, c2w, c2i]
        theta: (K, 9) parameter draws, columns ordered as PARAM_ORDER

    Returns:
        (n, K) float64 array of predicted rates
    """
    cols = [jnp.asarray(X[:, j : j + 1]) for j in range(6)]
    p_i = calculate_partial_pressures(cols[1], cols[2], cols[3], cols[4], cols[5])

    r = kinetics_model(
        p_i,
        cols[0],
        *[jnp.asarray(theta[:, j]) for j in range(8)],
    )
    # kinetics_model runs in float32 (jax_enable_x64 is off repo-wide); the measured
    # float32-vs-float64 discrepancy on r is <2e-5 relative, negligible against a ~10%
    # noise model. The log-density arithmetic below is done in float64, where the
    # precision actually matters.
    return np.asarray(r, dtype=np.float64)


def eig_pce(
    theta_outer: np.ndarray,
    theta_inner: np.ndarray,
    x_new: np.ndarray,
    eps: np.ndarray,
) -> dict:
    """
    Parametric expected information gain -- used in this script ONLY for the data
    processing inequality check, never as the objective. See HSA_K_001 for the
    parametric acquisition itself.

    Estimates I(theta ; y | x, D) via the contrastive (PCE) lower bound

        I_pce = (1/N) sum_i [ log p(y_i|theta_i) - log( mean over {theta_i} u {theta_m} of p(y_i|.) ) ]

    with y_i ~ p(y | theta_i, x). The plain NMC form (contrastive set excluding
    theta_i) is an upper bound; the gap between them is the Monte Carlo error.

    The outer and inner draws MUST be disjoint -- if theta_i were already in the
    contrastive set the two forms would coincide and the bound would be destroyed.

    Args:
        theta_outer: (N, 9) outer draws
        theta_inner: (M, 9) contrastive draws, disjoint from theta_outer
        x_new: Design point (6,) [T, p, c2h, c2m, c2w, c2i]
        eps: (N,) standard normal draws for the simulated observations

    Returns:
        dict with pce (objective), sem, nmc_upper, saturated_fraction, r_min, r_max
    """
    n_outer = theta_outer.shape[0]
    n_inner = theta_inner.shape[0]

    theta_all = np.concatenate([theta_outer, theta_inner], axis=0)
    r_all = rates_at(x_new, theta_all)
    s_all = theta_all[:, 8] * np.abs(r_all) + SIGMA_FLOOR

    if not np.all(np.isfinite(r_all)):
        logger.warning(
            f"Non-finite rate at design {np.asarray(x_new).tolist()}; scoring it as uninformative."
        )
        return {
            "pce": 0.0,
            "sem": 0.0,
            "nmc_upper": 0.0,
            "saturated_fraction": 0.0,
            "r_min": float("nan"),
            "r_max": float("nan"),
        }

    r_o, s_o = r_all[:n_outer], s_all[:n_outer]
    r_i, s_i = r_all[n_outer:], s_all[n_outer:]

    # Simulate one observation per outer draw from that draw's own noise model.
    # y_i - r_i = s_i * eps_i, so the self log-density needs no subtraction.
    log_self = -0.5 * np.log(2.0 * np.pi * s_o**2) - 0.5 * eps**2
    y = r_o + s_o * eps

    # (N, M) contrastive log-densities: how well every other draw explains y_i.
    z = (y[:, None] - r_i[None, :]) / s_i[None, :]
    log_contrast = -0.5 * np.log(2.0 * np.pi * s_i[None, :] ** 2) - 0.5 * z**2

    # One reduction serves both bounds. Appending the self-term via logaddexp is
    # identical to logsumexp over the concatenation, at half the peak memory.
    lse_contrast = logsumexp(log_contrast, axis=1)

    # NMC (upper bound): contrastive set excludes theta_i.
    per_sample_nmc = log_self - lse_contrast + np.log(n_inner)

    # PCE (lower bound): contrastive set includes theta_i itself.
    per_sample_pce = (
        log_self - np.logaddexp(lse_contrast, log_self) + np.log(n_inner + 1)
    )

    ceiling = np.log(n_inner + 1)
    saturated = float(np.mean(per_sample_pce > ceiling - SATURATION_TOL))

    return {
        "pce": float(per_sample_pce.mean()),
        # Ax's second raw_data element is a standard error, not a standard deviation.
        "sem": float(per_sample_pce.std(ddof=1) / np.sqrt(n_outer)),
        "nmc_upper": float(per_sample_nmc.mean()),
        "saturated_fraction": saturated,
        "r_min": float(r_all.min()),
        "r_max": float(r_all.max()),
    }


def generate_test_grid(n_points: int = 20, seed: int = TEST_GRID_SEED) -> np.ndarray:
    """
    Latin-hypercube test grid over the design box.

    The grid defines the region whose predictions the criterion cares about, so it
    is held FIXED for the whole campaign and across seeds -- otherwise the objective
    would mean something different from one run to the next. Seeded, unlike the
    unseeded sampler in HSA_H_003.

    Args:
        n_points: Number of test points K
        seed: PRNG seed for the Latin-hypercube sampler

    Returns:
        (K, 6) float64 array of test conditions
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    sample = sampler.random(n=n_points)
    lower = np.array([b[0] for b in DESIGN_BOUNDS], dtype=np.float64)
    upper = np.array([b[1] for b in DESIGN_BOUNDS], dtype=np.float64)
    return np.asarray(qmc.scale(sample, lower, upper), dtype=np.float64)


def test_grid_loglik(
    x_test: np.ndarray,
    theta_outer: np.ndarray,
    theta_inner: np.ndarray,
    eps_test: np.ndarray,
) -> dict:
    """
    Log-density of simulated test-grid predictions under every contrastive draw.

    Returns B with

        B[i, m] = sum_k log p( ytilde_ik | theta_m, xtest_k )

    i.e. the JOINT density over the whole grid for draw m. Summing over test points
    INSIDE this matrix -- before the mixture average taken in eig_predictive -- is
    exactly what makes the criterion joint over the grid rather than K separate
    marginals. The accumulation runs over k into one (N, M) buffer, so the
    (N, M, K) intermediate is never formed.

    None of this depends on the candidate, so it is computed once per design
    iteration and reused for every Ax trial.

    Args:
        x_test: (K, 6) test grid
        theta_outer: (N, 9) outer draws
        theta_inner: (M, 9) contrastive draws, disjoint from theta_outer
        eps_test: (N, K) standard normal draws for the simulated test observations

    Returns:
        dict with B (N, M) and its row-wise logsumexp over the full and half
        contrastive sets, both reused by every candidate
    """
    n_outer = theta_outer.shape[0]
    n_inner = theta_inner.shape[0]

    r_out = rates_grid(x_test, theta_outer)  # (K, N)
    r_in = rates_grid(x_test, theta_inner)  # (K, M)
    s_out = theta_outer[None, :, 8] * np.abs(r_out) + SIGMA_FLOOR
    s_in = theta_inner[None, :, 8] * np.abs(r_in) + SIGMA_FLOOR

    B = np.zeros((n_outer, n_inner), dtype=np.float64)
    for k in range(x_test.shape[0]):
        # Simulated observation at test point k, drawn from each outer draw's own
        # noise model -- the same sigma_r * |r| + floor used for real observations.
        y_k = r_out[k] + s_out[k] * eps_test[:, k]
        z = (y_k[:, None] - r_in[k][None, :]) / s_in[k][None, :]
        B += -0.5 * np.log(2.0 * np.pi * s_in[k][None, :] ** 2) - 0.5 * z**2

    half = n_inner // 2
    return {
        "B": B,
        "lse": logsumexp(B, axis=1),
        "lse_half": logsumexp(B[:, :half], axis=1),
        "half": half,
    }


def eig_predictive(
    theta_outer: np.ndarray,
    theta_inner: np.ndarray,
    x_new: np.ndarray,
    eps: np.ndarray,
    grid: dict,
) -> dict:
    """
    Inference-free estimate of the predictive expected information gain.

    Estimates I(ytilde ; y | x, xtest, D): how much an observation at the candidate
    tells you about the joint predictions on the test grid. Because y and ytilde are
    conditionally independent given theta,

        I = E[ log p(y, ytilde) - log p(y) - log p(ytilde) ]

    with every density a mixture average over the contrastive draws:

        U = mean_i[ lse_m(A + B) - lse_m(A) - lse_m(B) + log M ]

    There is no self-term -- this is a mutual information between two observables,
    so the diagonal never appears and K_001's PCE bound does not transfer.
    Convergence in M is reported instead as the shift when the contrastive set is
    halved.

    Args:
        theta_outer: (N, 9) outer draws
        theta_inner: (M, 9) contrastive draws, disjoint from theta_outer
        x_new: Design point (6,) [T, p, c2h, c2m, c2w, c2i]
        eps: (N,) standard normal draws for the simulated candidate observation
        grid: output of test_grid_loglik, reused across candidates

    Returns:
        dict with eig (objective), sem, conv_gap, r_min, r_max
    """
    n_outer = theta_outer.shape[0]
    n_inner = theta_inner.shape[0]

    theta_all = np.concatenate([theta_outer, theta_inner], axis=0)
    r_all = rates_at(x_new, theta_all)
    s_all = theta_all[:, 8] * np.abs(r_all) + SIGMA_FLOOR

    if not np.all(np.isfinite(r_all)):
        logger.warning(
            f"Non-finite rate at design {np.asarray(x_new).tolist()}; scoring it as uninformative."
        )
        return {
            "eig": 0.0,
            "sem": 0.0,
            "conv_gap": 0.0,
            "r_min": float("nan"),
            "r_max": float("nan"),
        }

    r_o, s_o = r_all[:n_outer], s_all[:n_outer]
    r_i, s_i = r_all[n_outer:], s_all[n_outer:]

    # Candidate observation simulated from each outer draw, then scored against
    # every contrastive draw: A[i, m] = log p(y_i | theta_m, x).
    y = r_o + s_o * eps
    z = (y[:, None] - r_i[None, :]) / s_i[None, :]
    A = -0.5 * np.log(2.0 * np.pi * s_i[None, :] ** 2) - 0.5 * z**2

    B = grid["B"]
    per_sample = (
        logsumexp(A + B, axis=1) - logsumexp(A, axis=1) - grid["lse"] + np.log(n_inner)
    )

    # Convergence in M: the same estimator on half the contrastive draws. A gap
    # comparable to the value itself means M is too small.
    half = grid["half"]
    per_half = (
        logsumexp(A[:, :half] + B[:, :half], axis=1)
        - logsumexp(A[:, :half], axis=1)
        - grid["lse_half"]
        + np.log(half)
    )

    return {
        "eig": float(per_sample.mean()),
        # Ax's second raw_data element is a standard error, not a standard deviation.
        "sem": float(per_sample.std(ddof=1) / np.sqrt(n_outer)),
        "conv_gap": float(abs(per_sample.mean() - per_half.mean())),
        "r_min": float(r_all.min()),
        "r_max": float(r_all.max()),
    }


def find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    idata: xarray.DataTree,
    x_test: np.ndarray,
    n_outer: int,
    n_inner: int,
    key: jax.Array,
    filename: Path,
):
    """
    Optimize the design point that maximises the predictive expected information gain.

    The parameter draws and the observation noise are drawn ONCE here and reused for
    every candidate in this design iteration (common random numbers). That cancels the
    shared Monte Carlo noise in the difference between candidates, which is what the
    Bayesian optimizer actually ranks.

    Args:
        n_iterations: Number of optimization iterations
        n_trials_per_iter: Number of candidates proposed per iteration
        idata: Current posterior samples
        x_test: (K, 6) fixed test grid the predictions are scored on
        n_outer: Outer Monte Carlo draws N
        n_inner: Contrastive draws M
        key: JAX PRNG key for this design iteration
        filename: Path to save optimization results

    Returns:
        Tuple of (best_parameters, expected_KLD, index, name, best_diagnostics)
    """
    key_theta, key_eps, key_test = jax.random.split(key, 3)
    theta = draw_theta(idata, n_outer + n_inner, key_theta)
    theta_outer = theta[:n_outer]
    theta_inner = theta[n_outer:]
    eps = np.asarray(jax.random.normal(key_eps, (n_outer,)), dtype=np.float64)

    # The test-grid term is independent of the candidate, so it is built once here
    # and reused by every Ax trial in this design iteration.
    eps_test = np.asarray(
        jax.random.normal(key_test, (n_outer, x_test.shape[0])), dtype=np.float64
    )
    _t_grid = time.perf_counter()
    grid = test_grid_loglik(x_test, theta_outer, theta_inner, eps_test)
    t_grid = time.perf_counter() - _t_grid

    client = Client()
    parameters = [
        RangeParameterConfig(
            name=name,
            parameter_type="float",
            bounds=bounds,
            **({"scaling": scaling} if scaling is not None else {}),
        )
        for name, bounds, scaling in DESIGN_SPEC
    ]
    client.configure_experiment(
        parameters=parameters,
    )
    # The metric keeps the name "KLD" so saved ax_client_*.json files stay readable
    # by the existing analysis tooling. The value is now the joint EIG.
    client.configure_optimization(objective="KLD")

    t_ax = 0.0
    t_eig = 0.0
    for _ in range(n_iterations):
        _t = time.perf_counter()
        trials = client.get_next_trials(max_trials=n_trials_per_iter)
        t_ax += time.perf_counter() - _t
        for trial_idx, trial_parameters in trials.items():
            candidate = np.array(
                [
                    trial_parameters["T"],
                    trial_parameters["p"],
                    trial_parameters["c2h"],
                    trial_parameters["c2m"],
                    trial_parameters["c2w"],
                    trial_parameters["c2i"],
                ],
                dtype=np.float64,
            )
            _t = time.perf_counter()
            result = eig_predictive(theta_outer, theta_inner, candidate, eps, grid)
            t_eig += time.perf_counter() - _t
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"KLD": (result["eig"], result["sem"])},
            )
            logger.debug(
                f"Trial {trial_idx} predictive EIG {result['eig']:.4f} +/- "
                f"{result['sem']:.4f} (conv gap {result['conv_gap']:.4f})"
            )

    logger.info(
        f"  acquisition timing: Ax surrogate {t_ax:.1f}s, EIG estimator {t_eig:.1f}s, "
        f"test grid {t_grid:.1f}s once "
        f"({n_iterations * n_trials_per_iter} candidates)"
    )
    client.summarize()
    client.save_to_json_file(str(filename))
    best_parameters, expected_KLD, index, name = client.get_best_parameterization()

    # Re-evaluate at the winner to recover its diagnostics. With the same common
    # random numbers this reproduces the trial value exactly, and costs milliseconds.
    best_candidate = np.array(
        [
            best_parameters["T"],
            best_parameters["p"],
            best_parameters["c2h"],
            best_parameters["c2m"],
            best_parameters["c2w"],
            best_parameters["c2i"],
        ],
        dtype=np.float64,
    )
    best_diagnostics = eig_predictive(
        theta_outer, theta_inner, best_candidate, eps, grid
    )

    # Data processing inequality: ytilde is a noisy function of theta, so the
    # predictive gain can never exceed the parametric gain. A violation is proof the
    # estimate is an artifact rather than a judgement call. Costs one extra
    # evaluation per design iteration.
    parametric = eig_pce(theta_outer, theta_inner, best_candidate, eps)
    best_diagnostics["parametric_eig"] = parametric["pce"]
    if best_diagnostics["eig"] > parametric["nmc_upper"]:
        logger.warning(
            f"DPI VIOLATION: predictive EIG {best_diagnostics['eig']:.4f} exceeds the "
            f"parametric upper bound {parametric['nmc_upper']:.4f}. The predictive "
            f"estimate is not trustworthy at this candidate."
        )

    if best_diagnostics["conv_gap"] > 0.1 * abs(best_diagnostics["eig"]):
        logger.warning(
            f"Predictive EIG shifts by {best_diagnostics['conv_gap']:.4f} when the "
            f"contrastive set is halved; consider raising --n-inner above {n_inner}."
        )

    if best_diagnostics["r_min"] < 0.0:
        logger.warning(
            "Selected design lies in the reverse-reaction region (driving force < 0); "
            f"rates span [{best_diagnostics['r_min']:.3e}, {best_diagnostics['r_max']:.3e}]."
        )
    return best_parameters, expected_KLD, index, name, best_diagnostics


def run_single_workflow(args, seed: int, exp_id: str, base_save_path: Path):
    seed_save_path = base_save_path / f"seed{seed:03d}"
    seed_save_path.mkdir(parents=True, exist_ok=True)

    n_outer = args.n_outer
    n_inner = args.n_inner
    n_iterations = args.iterations
    n_opt_iterations = args.optiterations
    n_trials_per_iter = args.trialsperiteration
    n_test_points = args.ntestpoints

    logger.info("=" * 70)
    logger.info(f"BAYESIAN OED - PREDICTIVE EIG SEQUENTIAL LOOP (Seed: {seed})")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Outer MC draws: {n_outer}")
    logger.info(f"  Contrastive draws: {n_inner}")
    logger.info(f"  Test-grid points: {n_test_points} (seed {TEST_GRID_SEED}, fixed)")
    logger.info(f"  Design iterations: {n_iterations}")
    logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Trials per iteration: {n_trials_per_iter}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    # Fixed for the whole campaign: the grid defines the region of interest, so it
    # must not change between design iterations or seeds.
    x_test = generate_test_grid(n_points=n_test_points)

    logger.info("\n[STEP 0] Sampling from prior and finding initial experiment...")
    prior_idata = sample_prior(num_samples=args.prior_samples)
    # Alias the prior into the posterior group so iteration 0 uses the same code path.
    prior_idata.posterior = prior_idata.prior

    best_parameters, expected_KLD, _, _, best_diag = find_opt_candidate(
        n_iterations=n_opt_iterations,
        n_trials_per_iter=n_trials_per_iter,
        idata=prior_idata,
        x_test=x_test,
        n_outer=n_outer,
        n_inner=n_inner,
        key=jax.random.PRNGKey(seed * 1000),
        filename=seed_save_path / "ax_client_000.json",
    )

    # Initialize with first proposal
    proposals = jnp.array(
        [
            [
                best_parameters["T"],
                best_parameters["p"],
                best_parameters["c2h"],
                best_parameters["c2m"],
                best_parameters["c2w"],
                best_parameters["c2i"],
            ]
        ],
    )

    logger.info(
        f"Initial proposal: T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar"
    )
    logger.info(f"Expected EIG: {expected_KLD['KLD'][0]:.4e}\n")

    trials_sum = []

    for i in range(1, n_iterations + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[ITERATION {i}/{n_iterations}] [Seed: {seed}]")
        logger.info(f"{'='*70}")

        # Extract experimental conditions
        T = proposals[:, 0:1]
        p = proposals[:, 1:2]
        c2h = proposals[:, 2:3]
        c2m = proposals[:, 3:4]
        c2w = proposals[:, 4:5]
        c2i = proposals[:, 5:6]

        logger.info(f"Conducting {proposals.shape[0]} experiments...")
        T, p_i, r_obs = generate_synthetic_data(
            T, p, c2h, c2m, c2w, c2i, seed=seed, rel_noise_std=0.1
        )
        logger.debug(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

        # Run Bayesian inference on the real data. This is the only MCMC left.
        logger.info("Running MCMC inference...")
        _t_mcmc = time.perf_counter()
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs,
            num_samples=4000,
            num_warmup=4000,
            num_chains=8,
        )

        logger.info(f"  MCMC wall time: {time.perf_counter() - _t_mcmc:.1f}s")

        trace_filename = seed_save_path / f"trace_{i:03d}.nc"
        idata.to_netcdf(trace_filename)
        logger.info(f"Saved trace to: {trace_filename.name}")

        logger.info(f"Optimizing next experiment (predictive EIG maximization)...")
        best_parameters, expected_KLD, _, _, best_diag = find_opt_candidate(
            n_iterations=n_opt_iterations,
            n_trials_per_iter=n_trials_per_iter,
            idata=idata,
            x_test=x_test,
            n_outer=n_outer,
            n_inner=n_inner,
            key=jax.random.PRNGKey(seed * 1000 + i),
            filename=seed_save_path / f"ax_client_{i:03d}.json",
        )

        # Add new proposal to list
        new_proposal = jnp.array(
            [
                [
                    best_parameters["T"],
                    best_parameters["p"],
                    best_parameters["c2h"],
                    best_parameters["c2m"],
                    best_parameters["c2w"],
                    best_parameters["c2i"],
                ]
            ],
        )
        proposals = jnp.concatenate([proposals, new_proposal], axis=0)

        logger.info(f"Next optimal experiment:")
        logger.info(
            f"  T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar,"
            f"c2h={best_parameters['c2h']:.2f}, c2m={best_parameters['c2m']:.1f}, "
            f"c2w={best_parameters['c2w']:.1f}, c2i={best_parameters['c2i']:.2f}"
        )
        logger.info(f"  Expected EIG: {expected_KLD['KLD'][0]:.4e}")

        # Store iteration summary
        trial_summary = {
            "iteration": i,
            "seed": seed,
            "observed": {
                "x": proposals[:-1].tolist(),  # All proposals except the new one
                "y": r_obs.tolist(),
            },
            "next_proposal": {
                "x": new_proposal.tolist()[0],
                "expected_KLD": float(expected_KLD["KLD"][0]),
                "KLD_std": (
                    float(expected_KLD["KLD"][1])
                    if len(expected_KLD["KLD"]) > 1
                    else None
                ),
                # Diagnostics only -- never an alternative objective.
                "conv_gap": best_diag["conv_gap"],
                "parametric_eig": best_diag["parametric_eig"],
            },
            "trace_path": str(trace_filename),
        }
        trials_sum.append(trial_summary)

        seed_exp_summary = {
            "experiment_id": exp_id,
            "seed": seed,
            "root_file": str(Path(__file__)),
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "trials": trials_sum,
        }
        summary_file = seed_save_path / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(seed_exp_summary, f, indent=2)

        logger.info(f"Updated experiment summary: {summary_file.name}")

    logger.info(f"\n{'='*70}")
    logger.info(f"SEQUENTIAL DESIGN COMPLETE (Seed: {seed})")
    logger.info(f"{'='*70}")
    logger.info(f"Total experiments conducted: {proposals.shape[0]}")
    logger.info(f"Results saved to: {seed_save_path}")
    logger.info(f"{'='*70}\n")

    # Drop large per-seed references before the next seed starts.
    del idata, prior_idata, proposals, r_obs, trials_sum
    gc.collect()
    clear_caches = getattr(jax, "clear_caches", None)
    if callable(clear_caches):
        clear_caches()

    return seed_save_path


def main(args):
    if args.prior_samples < args.n_outer + args.n_inner:
        parser.error(
            f"--prior-samples ({args.prior_samples}) must be at least "
            f"--n-outer + --n-inner ({args.n_outer + args.n_inner})."
        )

    base_seed = args.seed
    n_seeds = args.n_seeds

    global EXP_ID
    EXP_ID = namer.generate()

    base_save_path = args.datadir / f"{EXP_ID}"
    base_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting multi-seed workflow")
    logger.info(f"  Base seed: {base_seed}")
    logger.info(f"  Number of seeds: {n_seeds}")
    logger.info(f"  Experiment ID: {EXP_ID}")
    logger.info(f"  Base path: {base_save_path}")

    all_save_paths = []
    seeds_used = []

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        seeds_used.append(seed)
        logger.info(f"\n{'#'*70}")
        logger.info(f"# STARTING SEED {seed_offset + 1}/{n_seeds}: SEED={seed}")
        logger.info(f"{'#'*70}\n")

        seed_save_path = run_single_workflow(args, seed, EXP_ID, base_save_path)
        all_save_paths.append(seed_save_path)
        logger.info(f"Completed workflow with seed {seed}")

        gc.collect()
        clear_caches = getattr(jax, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()

    master_summary = {
        "experiment_id": EXP_ID,
        "base_seed": base_seed,
        "n_seeds": n_seeds,
        "seeds_used": seeds_used,
        "seed_paths": {s: str(p) for s, p in zip(seeds_used, all_save_paths)},
        "timestamp": str(datetime.datetime.now()),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    master_summary_file = base_save_path / "multi_seed_summary.json"
    with open(master_summary_file, "w") as f:
        json.dump(master_summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("ALL WORKFLOWS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Master summary saved to: {master_summary_file}")
    logger.info(f"Experiment ID: {EXP_ID}")
    logger.info(f"Seeds used: {seeds_used}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
