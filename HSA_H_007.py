"""
LHS experimental plan (Numpyro Version) - Multi-Seed Support

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

from scipy.stats import qmc
from scipy.integrate import simpson
import xarray
import namer
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
    "--iterations",
    type=int,
    default=32,
    help="Number of points to additionally sample in the experiment",
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


def LHS_design(n_samples: int, seed: int):
    # Define the ranges
    T_range = (450.0, 600.0)
    p_range = (3.0, 9.0)
    c2h_range = (0.2, 4.0)
    c2m_range = (0.5, 1000.0)  # Logarithmic
    c2w_range = (0.5, 1000.0)  # Logarithmic
    c2i_range = (0.05, 1.0)

    n_params = 6  # Number of parameters

    # Generate LHS samples in [0, 1]
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale each parameter to its range
    scaled_samples = np.zeros_like(samples)
    scaled_samples[:, 0] = T_range[0] + samples[:, 0] * (T_range[1] - T_range[0])
    scaled_samples[:, 1] = p_range[0] + samples[:, 1] * (p_range[1] - p_range[0])
    scaled_samples[:, 2] = c2h_range[0] + samples[:, 2] * (c2h_range[1] - c2h_range[0])

    # Logarithmic scaling for c2m and c2w
    scaled_samples[:, 3] = c2m_range[0] * (c2m_range[1] / c2m_range[0]) ** samples[:, 3]
    scaled_samples[:, 4] = c2w_range[0] * (c2w_range[1] / c2w_range[0]) ** samples[:, 4]

    scaled_samples[:, 5] = c2i_range[0] + samples[:, 5] * (c2i_range[1] - c2i_range[0])

    proposals = jnp.array(scaled_samples)
    return proposals


def run_single_workflow(n_iterations, seed: int, exp_id: str, base_save_path: Path):
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

    logger.info("=" * 70)
    logger.info(f"KOSCHANY DESIGN - SEQUENTIAL LOOP (Seed: {seed})")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    # logger.info(f"  Workers: {num_workers}")
    # logger.info(f"  KL samples: {n_hyp_samples}")
    # logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Design iterations: {n_iterations}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    all_proposals = LHS_design(n_iterations, seed)

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

def _single_workflow(args):
    n_iterations, seed, exp_id, base_save_path = args
    seed_save_path = run_single_workflow(n_iterations, seed, exp_id, base_save_path)
    return seed_save_path

def main(args):
    """Run workflow for multiple seeds with shared EXP_ID."""
    base_seed = args.seed
    n_seeds = args.n_seeds
    n_iterations = args.iterations

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

        seed_save_path = run_single_workflow(n_iterations, seed, EXP_ID, base_save_path)
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
