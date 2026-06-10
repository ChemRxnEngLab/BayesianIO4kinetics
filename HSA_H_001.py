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
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
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
    default=42,
    help="Base random seed for reproducible workflow execution",
)
parser.add_argument(
    "--n-seeds",
    type=int,
    default=1,
    help="Number of different seeds to run (for ensemble studies)",
)


def kl_marginal(P_samples, Q_samples, eps=1e-12):
    kde_P = gaussian_kde(P_samples)
    kde_Q = gaussian_kde(Q_samples)

    grid = np.linspace(
        min(P_samples.min(), Q_samples.min()),
        max(P_samples.max(), Q_samples.max()),
        1000,
    )

    p_vals = kde_P(grid) + eps
    q_vals = kde_Q(grid) + eps

    kld = simpson(p_vals * np.log(p_vals / q_vals), grid)
    return kld


def kl_divergence_between_models(
    idata_new,
    idata_old,
    key_new="posterior",
    key_old="posterior",
    n_samples=1000,
    var_names=None,
):
    if var_names is None:
        var_names = list(idata_new[key_new].data_vars.keys())

    old_samples = az.extract(
        idata_old,
        group=key_old,
        num_samples=n_samples,
        var_names=var_names,
    )
    new_samples = az.extract(
        idata_new,
        group=key_new,
        num_samples=n_samples,
        var_names=var_names,
    )
    klds = []
    for var in old_samples.data_vars:
        P_samples = new_samples[var].values.flatten()
        Q_samples = old_samples[var].values.flatten()
        kl_div = kl_marginal(P_samples, Q_samples)
        klds.append(kl_div)
    return np.array(klds)


def KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    use_vi: bool = True,
):
    """
    Compute KL divergence between posterior distributions before and after adding hypothetical data.

    KL(P_new || P_old) = E_P_new[log P_new(theta) - log P_old(theta)]

    Args:
        x_old: Current experimental conditions (n_current, 5) [T, p, c2h, f, c2i]
        y_old: Current observations (n_current, 1)
        x_new: New experimental condition to test (5,) [T, p, c2h, f, c2i]
        idata_old: Current posterior samples (ArviZ InferenceData)
        n_hyp_samples: Number of hypothetical observations to average over
        use_vi: If True, use variational inference (faster); if False, use MCMC

    Returns:
        xarray.DataArray of KL divergences with dimensions (hyp_sample, parameter)
    """
    T_new = x_new[0:1].reshape(-1, 1)
    p_new = x_new[1:2].reshape(-1, 1)
    c2h_new = x_new[2:3].reshape(-1, 1)
    c2m_new = x_new[3:4].reshape(-1, 1)
    c2w_new = x_new[4:5].reshape(-1, 1)
    c2i_new = x_new[5:6].reshape(-1, 1)

    p_i_new = calculate_partial_pressures(p_new, c2h_new, c2m_new, c2w_new, c2i_new)

    y_new_samples = predict(
        idata_old, T_new, p_i_new, key="posterior", num_samples=n_hyp_samples
    ).reshape(-1, 1)
    logger.debug(f"Generated {n_hyp_samples} hypothetical observations at new condition")

    klds = xarray.DataArray(
        dims=["hyp_sample", "parameter"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "parameter": [
                "k0", "Ea_k", "A_oh", "dH_oh",
                "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r",
            ],
        },
        data=np.zeros((n_hyp_samples, 9)),
    )

    for i in range(n_hyp_samples):
        logger.debug(f"Processing hypothetical sample {i+1}/{n_hyp_samples}")

        y_hyp = jnp.array(y_new_samples[i].reshape(1, 1))
        x_augmented = jnp.vstack([x_old, x_new.reshape(1, -1)])
        y_augmented = jnp.vstack([y_old, y_hyp])

        T_aug = x_augmented[:, 0:1]
        p_aug = x_augmented[:, 1:2]
        c2h_aug = x_augmented[:, 2:3]
        c2m_aug = x_augmented[:, 3:4]
        c2w_aug = x_augmented[:, 4:5]
        c2i_aug = x_augmented[:, 5:6]

        p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, c2m_aug, c2w_aug, c2i_aug)

        idata_new = run_inference(
            T_aug, p_i_aug, y_augmented,
            num_samples=4000, num_warmup=1000, num_chains=2,
        )

        klds_i = kl_divergence_between_models(
            idata_new, idata_old, key_new="posterior", key_old="posterior",
            n_samples=1000,
            var_names=[
                "k0", "Ea_k", "A_oh", "dH_oh",
                "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r",
            ],
        )
        klds.loc[i, :] = klds_i

    return klds


def _single_kl_worker(args):
    """Worker function for computing KL divergence for a single hypothetical observation."""
    x_old, y_old, x_new, idata_old, y_hyp, use_vi = args

    x_augmented = jnp.vstack([x_old, x_new.reshape(1, -1)])
    y_augmented = jnp.vstack([y_old, y_hyp])

    T_aug = x_augmented[:, 0:1]
    p_aug = x_augmented[:, 1:2]
    c2h_aug = x_augmented[:, 2:3]
    c2m_aug = x_augmented[:, 3:4]
    c2w_aug = x_augmented[:, 4:5]
    c2i_aug = x_augmented[:, 5:6]

    p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, c2m_aug, c2w_aug, c2i_aug)

    idata_new = run_inference(
        T_aug, p_i_aug, y_augmented,
        num_samples=4000, num_warmup=1000, num_chains=2,
    )

    klds_i = kl_divergence_between_models(
        idata_new, idata_old, key_new="posterior", key_old="posterior",
        n_samples=1000,
        var_names=[
            "k0", "Ea_k", "A_oh", "dH_oh",
            "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r",
        ],
    )
    return klds_i


def _candidate_kl_worker(args):
    """Top-level worker for parallelizing over candidate proposals."""
    x_old, y_old, x_new, idata_old, n_hyp_samples, n_cores, use_vi = args
    return mp_KL_divergence(
        x_old=x_old, y_old=y_old, x_new=x_new.squeeze(),
        idata_old=idata_old, n_hyp_samples=n_hyp_samples,
        n_cores=n_cores, use_vi=use_vi,
    )


def mp_KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    n_cores: int = 4,
    use_vi: bool = True,
):
    """
    Compute KL divergence in parallel across multiple hypothetical observations.
    """
    T_new = x_new[0:1].reshape(-1, 1)
    p_new = x_new[1:2].reshape(-1, 1)
    c2h_new = x_new[2:3].reshape(-1, 1)
    c2m_new = x_new[3:4].reshape(-1, 1)
    c2w_new = x_new[4:5].reshape(-1, 1)
    c2i_new = x_new[5:6].reshape(-1, 1)

    p_i_new = calculate_partial_pressures(p_new, c2h_new, c2m_new, c2w_new, c2i_new)

    y_new_samples = predict(
        idata_old, T_new, p_i_new, key="posterior", num_samples=n_hyp_samples
    ).reshape(-1, 1)
    logger.debug(f"Generated {n_hyp_samples} hypothetical observations at new condition")

    args_list = [
        (x_old, y_old, x_new, idata_old, jnp.array(y_new_samples[i].reshape(1, 1)), use_vi)
        for i in range(n_hyp_samples)
    ]
    n_procs = min(n_cores, n_hyp_samples)
    with NestablePool(n_procs) as pool:
        results = pool.map(_single_kl_worker, args_list)
    logger.debug("Completed KL divergence computations in parallel")

    klds = xarray.DataArray(
        dims=["hyp_sample", "parameter"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "parameter": [
                "k0", "Ea_k", "A_oh", "dH_oh",
                "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r",
            ],
        },
        data=np.stack(results, axis=0),
    )
    return klds


def koschany_design():
    proposals = [
        # T
        [525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         450, 475, 505,
         450, 475, 505,
         450, 475, 505,
         450, 475, 505,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585,
         525, 540, 555, 585],
        # p
        [6, 6, 6, 6,
         6, 6, 6, 6,
         6, 6, 6, 6,
         6, 6, 6, 6,
         6, 6, 6, 6,
         6, 6, 6, 6,
         6, 6, 6, 6,
         4, 4, 4,
         4, 4, 4,
         4, 4, 4,
         4, 4, 4,
         9, 9, 9, 9,
         6, 6, 6, 6,
         3, 3, 3, 3,
         9, 9, 9, 9,
         6, 6, 6, 6,
         3, 3, 3, 3,
         9, 9, 9, 9,
         6, 6, 6, 6,
         3, 3, 3, 3],
        # C2H
        [0.2, 0.2, 0.2, 0.2,
         0.25, 0.25, 0.25, 0.25,
         0.4, 0.4, 0.4, 0.4,
         0.8, 0.8, 0.8, 0.8,
         1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0,
         4.0, 4.0, 4.0, 4.0,
         0.25, 0.25, 0.25,
         0.66, 0.66, 0.66,
         1.0, 1.0, 1.0,
         1.5, 1.5, 1.5,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25,
         0.25, 0.25, 0.25, 0.25],
        # C2M
        [1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         0.800, 0.800, 0.800, 0.800,
         0.800, 0.800, 0.800, 0.800,
         0.800, 0.800, 0.800, 0.800],
        # C2W
        [1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         1000.0, 1000.0, 1000.0,
         0.800, 0.800, 0.800, 0.800,
         0.800, 0.800, 0.800, 0.800,
         0.800, 0.800, 0.800, 0.800],
        # c2i
        [0.071, 0.071, 0.071, 0.071,
         0.2, 0.2, 0.2, 0.2,
         0.153, 0.153, 0.153, 0.153,
         0.364, 0.364, 0.364, 0.364,
         0.333, 0.333, 0.333, 0.333,
         0.286, 0.286, 0.286, 0.286,
         0.800, 0.800, 0.800, 0.800,
         0.2, 0.2, 0.2,
         0.4, 0.4, 0.4,
         0.5, 0.5, 0.5,
         0.6, 0.6, 0.6,
         0.2, 0.2, 0.2, 0.2,
         0.2, 0.2, 0.2, 0.2,
         0.2, 0.2, 0.2, 0.2,
         1.000, 1.000, 1.000, 1.000,
         1.000, 1.000, 1.000, 1.000,
         1.000, 1.000, 1.000, 1.000,
         1.000, 1.000, 1.000, 1.000],
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

        KLD_vals = mp_KL_divergence(
            x_old=proposals,
            y_old=r_obs,
            x_new=new_proposal.squeeze(),
            idata_old=prior_idata,
            n_hyp_samples=n_hyp_samples,
            n_cores=num_workers,
            use_vi=True,
        )
        mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
        std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()

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
        idata.to_netcdf(trace_filename)
        logger.info(f"Saved trace to: {trace_filename.name}")

        trial_summary = {
            "iteration": i,
            "seed": seed,
            "observed": {"x": proposals.tolist(), "y": r_obs.tolist()},
            "predicted_before_obs": {"expected_KLD": mean_KLD, "std_KLD": std_KLD},
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
    
    return seed_save_path


def main(args):
    """Run workflow for multiple seeds with shared EXP_ID."""
    base_seed = args.seed
    n_seeds = args.n_seeds
    
    global EXP_ID
    EXP_ID = namer.generate()
    
    base_save_path = Path.cwd() / "data" / f"{EXP_ID}"
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
