"""
Bayesian Optimal Experimental Design using an inference-free joint EIG (NumPyro) - Multi-Seed

This module implements sequential Bayesian optimal experimental design for kinetic
parameter estimation. The utility is the expected information gain

    U(x) = E_y[ KL( p(theta | y, x, D) || p(theta | D) ) ] = I(theta ; y | x, D)

which is a KL over the FULL parameter vector, not a sum of per-parameter marginals.

The estimator never refits the model. By Bayes' rule

    p(theta | y) / p(theta) = p(y | theta) / p(y)

so the posterior density cancels, and the expected KL becomes an expectation over the
joint p(theta, y). That joint is sampled by drawing theta from the CURRENT posterior
(the existing trace) and simulating y forward through the closed-form Normal likelihood
-- no MCMC per hypothetical observation. The result is inherently joint: log p(y|theta)
is evaluated at the whole vector and never factorises over parameters.

The objective is the contrastive (PCE) lower bound on the mutual information; the
plain NMC form is computed from the same matrix as an upper bound and logged as a
convergence diagnostic only. It is never used as an objective.

Each seed run:
- Uses the same EXP_ID
- Stores results in {datadir}/{EXP_ID}/seed{seed:03d}/
- Generates synthetic data with the specified seed

Key Features:
- Inference-free nested-Monte-Carlo estimate of the joint EIG
- PCE lower bound / NMC upper bound sandwich for Monte Carlo error readout
- Common random numbers across candidates within a design iteration
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


from HSA_Mo_B_011 import (
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
    help="Contrastive draws M for the EIG estimator (PCE ceiling is log(M+1) nats)",
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

    This is the only forward evaluation the estimator needs: one call covering all
    draws, since kinetics_model is already vectorised over the draw axis.

    Args:
        x_new: Design point (6,) [T, p, c2h, c2m, c2w, c2i]
        theta: (K, 9) parameter draws, columns ordered as PARAM_ORDER

    Returns:
        (K,) float64 array of predicted rates
    """
    x = jnp.asarray(x_new)
    T_new = x[0:1].reshape(-1, 1)
    p_new = x[1:2].reshape(-1, 1)
    c2h_new = x[2:3].reshape(-1, 1)
    c2m_new = x[3:4].reshape(-1, 1)
    c2w_new = x[4:5].reshape(-1, 1)
    c2i_new = x[5:6].reshape(-1, 1)

    p_i_new = calculate_partial_pressures(p_new, c2h_new, c2m_new, c2w_new, c2i_new)

    r = kinetics_model(
        p_i_new,
        T_new,
        *[jnp.asarray(theta[:, j]) for j in range(8)],
    )
    # kinetics_model runs in float32 (jax_enable_x64 is off repo-wide); the measured
    # float32-vs-float64 discrepancy on r is <2e-5 relative, negligible against a ~10%
    # noise model. The log-density arithmetic below is done in float64, where the
    # precision actually matters.
    return np.asarray(r, dtype=np.float64).ravel()


def eig_pce(
    theta_outer: np.ndarray,
    theta_inner: np.ndarray,
    x_new: np.ndarray,
    eps: np.ndarray,
) -> dict:
    """
    Inference-free estimate of the joint expected information gain at one design point.

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


def find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    idata: xarray.DataTree,
    n_outer: int,
    n_inner: int,
    key: jax.Array,
    filename: Path,
):
    """
    Optimize the design point that maximises the joint expected information gain.

    The parameter draws and the observation noise are drawn ONCE here and reused for
    every candidate in this design iteration (common random numbers). That cancels the
    shared Monte Carlo noise in the difference between candidates, which is what the
    Bayesian optimizer actually ranks.

    Args:
        n_iterations: Number of optimization iterations
        n_trials_per_iter: Number of candidates proposed per iteration
        idata: Current posterior samples
        n_outer: Outer Monte Carlo draws N
        n_inner: Contrastive draws M
        key: JAX PRNG key for this design iteration
        filename: Path to save optimization results

    Returns:
        Tuple of (best_parameters, expected_KLD, index, name, best_diagnostics)
    """
    key_theta, key_eps = jax.random.split(key)
    theta = draw_theta(idata, n_outer + n_inner, key_theta)
    theta_outer = theta[:n_outer]
    theta_inner = theta[n_outer:]
    eps = np.asarray(jax.random.normal(key_eps, (n_outer,)), dtype=np.float64)

    client = Client()
    parameters = [
        RangeParameterConfig(
            name="T",
            parameter_type="float",
            bounds=(450.0, 600.0),
        ),
        RangeParameterConfig(
            name="p",
            parameter_type="float",
            bounds=(3.0, 9.0),
        ),
        RangeParameterConfig(
            name="c2h",
            parameter_type="float",
            scaling="log",
            bounds=(0.2, 4.0),
        ),
        RangeParameterConfig(
            name="c2m",
            parameter_type="float",
            scaling="log",
            bounds=(0.5, 1000.0),
        ),
        RangeParameterConfig(
            name="c2w",
            parameter_type="float",
            scaling="log",
            bounds=(0.5, 1000.0),
        ),
        RangeParameterConfig(
            name="c2i",
            parameter_type="float",
            scaling="log",
            bounds=(0.05, 1.0),
        ),
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
            result = eig_pce(theta_outer, theta_inner, candidate, eps)
            t_eig += time.perf_counter() - _t
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"KLD": (result["pce"], result["sem"])},
            )
            logger.debug(
                f"Trial {trial_idx} EIG {result['pce']:.4f} +/- {result['sem']:.4f} "
                f"(upper {result['nmc_upper']:.4f}, sat {result['saturated_fraction']:.1%})"
            )

    logger.info(
        f"  acquisition timing: Ax surrogate {t_ax:.1f}s, EIG estimator {t_eig:.1f}s "
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
    best_diagnostics = eig_pce(theta_outer, theta_inner, best_candidate, eps)

    if best_diagnostics["r_min"] < 0.0:
        logger.warning(
            "Selected design lies in the reverse-reaction region (driving force < 0); "
            f"rates span [{best_diagnostics['r_min']:.3e}, {best_diagnostics['r_max']:.3e}]."
        )
    if best_diagnostics["saturated_fraction"] > 0.01:
        logger.warning(
            f"{best_diagnostics['saturated_fraction']:.1%} of outer samples saturated at the "
            f"log(M+1) ceiling; consider raising --n-inner above {n_inner}."
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

    logger.info("=" * 70)
    logger.info(
        f"BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP (Seed: {seed})"
    )
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Outer MC draws: {n_outer}")
    logger.info(
        f"  Contrastive draws: {n_inner} (ceiling {np.log(n_inner + 1):.2f} nats)"
    )
    logger.info(f"  Design iterations: {n_iterations}")
    logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Trials per iteration: {n_trials_per_iter}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    logger.info("\n[STEP 0] Sampling from prior and finding initial experiment...")
    prior_idata = sample_prior(num_samples=args.prior_samples)
    # Alias the prior into the posterior group so iteration 0 uses the same code path.
    prior_idata.posterior = prior_idata.prior

    best_parameters, expected_KLD, _, _, best_diag = find_opt_candidate(
        n_iterations=n_opt_iterations,
        n_trials_per_iter=n_trials_per_iter,
        idata=prior_idata,
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

        logger.info(f"Optimizing next experiment (joint EIG maximization)...")
        best_parameters, expected_KLD, _, _, best_diag = find_opt_candidate(
            n_iterations=n_opt_iterations,
            n_trials_per_iter=n_trials_per_iter,
            idata=idata,
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
                "eig_upper_bound": best_diag["nmc_upper"],
                "saturated_fraction": best_diag["saturated_fraction"],
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
