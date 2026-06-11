"""
Koshany experimental plan (Numpyro Version) - Multi-Seed Support

This module implements the original Koschany et al. experimental design with support
for running multiple workflows with different random seeds.

Each seed run:
- Uses the same EXP_ID
- Stores results in data/{EXP_ID}/seed{seed:03d}/
- Generates synthetic data with the specified seed

"""

import json
import argparse
import gc
from pathlib import Path
import datetime
import logging
import arviz as az
import numpy as np

from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import xarray
import namer
from ax import Client, RangeParameterConfig
import numpyro

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

numpyro.set_host_device_count(48)
import jax
import jax.numpy as jnp

from NestPools import NestablePool


from HSA_Mo_B_011_numpyro import (
    run_inference,
    predict,
    sample_prior,
    generate_synthetic_data,
    calculate_partial_pressures,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers",
    type=int,
    default=12,
    help="Number of workers for parallel processing KLD computations",
)
parser.add_argument(
    "--KLsamples",
    type=int,
    default=12,
    help="Number of hypothetical samples for KL divergence estimation",
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
    default=10,
    help="Number of optimization iterations for Ax",
)
parser.add_argument(
    "--trialsperiteration",
    type=int,
    default=2,
    help="Number of trials per optimization iteration for Ax",
)
parser.add_argument(
    "--seed",
    type=int,
    default=43,
    help="Base random seed for reproducible workflow execution",
)
parser.add_argument(
    "--n-seeds",
    type=int,
    default=1,
    help="Number of different seeds to run (for ensemble studies)",
)


def koschany_design():
    proposals = [
        # T
        [
            525,  # r61
            540,
            555,
            585,
            525,  # r62
            540,
            555,
            585,
            525,  # r63
            540,
            555,
            585,
            525,  # r64
            540,
            555,
            585,
            525,  # r65
            540,
            555,
            585,
            525,  # r66
            540,
            555,
            585,
            525,  # r67
            540,
            555,
            585,
            450,  # l41
            475,
            505,
            450,  # l42
            475,
            505,
            450,  # l43
            475,
            505,
            450,  # l44
            475,
            505,
            525,  # p91
            540,
            555,
            585,
            525,  # p62
            540,
            555,
            585,
            525,  # p33
            540,
            555,
            585,
            525,  # p9H2O1
            540,
            555,
            585,
            525,  # p6H2O2
            540,
            555,
            585,
            525,  # p3H2O3
            540,
            555,
            585,
        ],
        # p
        [
            6,  # r61
            6,
            6,
            6,
            6,  # r62
            6,
            6,
            6,
            6,  # r63
            6,
            6,
            6,
            6,  # r64
            6,
            6,
            6,
            6,  # r65
            6,
            6,
            6,
            6,  # r66
            6,
            6,
            6,
            6,  # r67
            6,
            6,
            6,
            4,  # l41
            4,
            4,
            4,  # l42
            4,
            4,
            4,  # l43
            4,
            4,
            4,  # l44
            4,
            4,
            9,  # p91
            9,
            9,
            9,
            6,  # p62
            6,
            6,
            6,
            3,  # p33
            3,
            3,
            3,
            9,  # p9H2O1
            9,
            9,
            9,
            6,  # p6H2O2
            6,
            6,
            6,
            3,  # p3H2O3
            3,
            3,
            3,
        ],
        # C2H
        [
            0.2,  # r61
            0.2,
            0.2,
            0.2,
            0.25,  # r62
            0.25,
            0.25,
            0.25,
            0.4,  # r63
            0.4,
            0.4,
            0.4,
            0.8,  # r64
            0.8,
            0.8,
            0.8,
            1.0,  # r65
            1.0,
            1.0,
            1.0,
            2.0,  # r66
            2.0,
            2.0,
            2.0,
            4.0,  # r67
            4.0,
            4.0,
            4.0,
            0.25,  # l41
            0.25,
            0.25,
            0.66,  # l42
            0.66,
            0.66,
            1.0,  # l43
            1.0,
            1.0,
            1.5,  # l44
            1.5,
            1.5,
            0.25,  # p91
            0.25,
            0.25,
            0.25,
            0.25,  # p62
            0.25,
            0.25,
            0.25,
            0.25,  # p33
            0.25,
            0.25,
            0.25,
            0.25,  # p9H2O1
            0.25,
            0.25,
            0.25,
            0.25,  # p6H2O2
            0.25,
            0.25,
            0.25,
            0.25,  # p3H2O3
            0.25,
            0.25,
            0.25,
        ],
        # C2M
        [
            1000.0,  # r61
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r62
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r63
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r64
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r65
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r66
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r67
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # l41
            1000.0,
            1000.0,
            1000.0,  # l42
            1000.0,
            1000.0,
            1000.0,  # l43
            1000.0,
            1000.0,
            1000.0,  # l44
            1000.0,
            1000.0,
            1000.0,  # p91
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # p62
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # p33
            1000.0,
            1000.0,
            1000.0,
            0.800,  # p9H2O1
            0.800,
            0.800,
            0.800,
            0.800,  # p6H2O2
            0.800,
            0.800,
            0.800,
            0.800,  # p3H2O3
            0.800,
            0.800,
            0.800,
        ],
        # C2W
        [
            1000.0,  # r61
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r62
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r63
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r64
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r65
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r66
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # r67
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # l41
            1000.0,
            1000.0,
            1000.0,  # l42
            1000.0,
            1000.0,
            1000.0,  # l43
            1000.0,
            1000.0,
            1000.0,  # l44
            1000.0,
            1000.0,
            1000.0,  # p91
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # p62
            1000.0,
            1000.0,
            1000.0,
            1000.0,  # p33
            1000.0,
            1000.0,
            1000.0,
            0.800,  # p9H2O1
            0.800,
            0.800,
            0.800,
            0.800,  # p6H2O2
            0.800,
            0.800,
            0.800,
            0.800,  # p3H2O3
            0.800,
            0.800,
            0.800,
        ],
        # c2i
        [
            0.071,  # r61
            0.071,
            0.071,
            0.071,
            0.2,  # r62
            0.2,
            0.2,
            0.2,
            0.153,  # r63
            0.153,
            0.153,
            0.153,
            0.364,  # r64
            0.364,
            0.364,
            0.364,
            0.333,  # r65
            0.333,
            0.333,
            0.333,
            0.286,  # r66
            0.286,
            0.286,
            0.286,
            0.800,  # r67
            0.800,
            0.800,
            0.800,
            0.2,  # l41
            0.2,
            0.2,
            0.4,  # l42
            0.4,
            0.4,
            0.5,  # l43
            0.5,
            0.5,
            0.6,  # l44
            0.6,
            0.6,
            0.2,  # p91
            0.2,
            0.2,
            0.2,
            0.2,  # p62
            0.2,
            0.2,
            0.2,
            0.2,  # p33
            0.2,
            0.2,
            0.2,
            1.000,  # p9H2O1
            1.000,
            1.000,
            1.000,
            1.000,  # p6H2O2
            1.000,
            1.000,
            1.000,
            1.000,  # p3H2O3
            1.000,
            1.000,
            1.000,
        ],
    ]
    return jnp.array(proposals).T


def run_single_workflow(args, seed: int, exp_id: str, base_save_path: Path):
    """Run the complete DOE workflow with a specific seed.

    Args:
        args: Parsed command-line arguments
        seed: Random seed for this workflow run
        exp_id: Shared experiment ID for all seeds
        base_save_path: Base path (data/{EXP_ID}/)

    Returns:
        Path to the seed-specific save directory
    """
    seed_save_path = base_save_path / f"seed{seed:03d}"
    seed_save_path.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers
    n_hyp_samples = args.KLsamples
    n_opt_iterations = args.optiterations
    n_trials_per_iter = args.trialsperiteration

    logger.info("=" * 70)
    logger.info(f"KOSCHANY DESIGN - SEQUENTIAL LOOP (Seed: {seed})")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  KL samples: {n_hyp_samples}")
    logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    all_proposals = koschany_design()
    n_iterations = all_proposals.shape[0]

    prior_idata = sample_prior(num_samples=4000)
    prior_idata.add_groups(posterior=prior_idata.prior)

    proposals = jnp.empty((0, 6))
    r_obs = jnp.empty((0, 1))
    trials_sum = []

    for i in range(0, n_iterations):
        logger.info(f"\n{'='*70}")
        logger.info(f"[ITERATION {i}/{n_iterations}] [Seed: {seed}]")
        logger.info(f"{'='*70}")

        new_proposal = all_proposals[i : i + 1, :]
        proposals = jnp.concatenate([proposals, new_proposal], axis=0)

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
        logger.info(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

        logger.info("Running MCMC inference...")
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs,
            num_samples=4000,
            num_warmup=4000,
            num_chains=8,
        )
        prior_idata = idata.copy()

        trace_filename = seed_save_path / f"trace_{i:03d}.nc"
        idata.to_netcdf(str(trace_filename))
        logger.info(f"Saved trace to: {trace_filename.name}")

        trial_summary = {
            "iteration": i,
            "seed": seed,
            "observed": {"x": proposals.tolist(), "y": r_obs.tolist()},
            "predicted_before_obs": {"expected_KLD": None, "std_KLD": None},
            "trace_path": str(trace_filename),
        }
        trials_sum.append(trial_summary)

        seed_exp_summary = {
            "experiment_id": exp_id,
            "seed": seed,
            "root_file": str(Path(__file__)),
            "args": vars(args),
            "trials": trials_sum,
        }
        summary_file = seed_save_path / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(seed_exp_summary, f, indent=2)

        logger.info(f"Updated experiment summary: {summary_file.name}")

    logger.info(f"\n{'='*70}")
    logger.info(f"FF DESIGN COMPLETE (Seed: {seed})")
    logger.info(f"{'='*70}")
    logger.info(f"Total experiments conducted: {proposals.shape[0]}")
    logger.info(f"Results saved to: {seed_save_path}")
    logger.info(f"{'='*70}\n")

    # Drop large per-seed references before the next seed starts.
    del idata, prior_idata, proposals, r_obs, trials_sum, all_proposals
    gc.collect()
    clear_caches = getattr(jax, "clear_caches", None)
    if callable(clear_caches):
        clear_caches()

    return seed_save_path


def main(args):
    """Run workflow for multiple seeds with shared EXP_ID."""
    base_seed = args.seed
    n_seeds = args.n_seeds

    global EXP_ID
    EXP_ID = namer.generate()

    base_save_path = Path("/lustre/work/ws/ws1/ul_qmj36-BIBO_workspace") / "data" / f"{EXP_ID}"
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
        "args": vars(args),
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
