"""
Bayesian Optimal Experimental Design using KL Divergence (Numpyro Version) - Multi-Seed Support

This module implements Bayesian optimal experimental design for kinetic parameter
estimation using KL divergence as the utility function with support for multiple
random seeds. Refactored to work with NumPyro probabilistic programming framework.

Each seed run:
- Uses the same EXP_ID
- Stores results in data/{EXP_ID}/seed{seed:03d}/
- Generates synthetic data with the specified seed

Key Features:
- Homoscedastic noise modeling
- TWO acquisition functions:
  1. Parametric KLD: KL divergence between parameter posteriors
  2. Predictive KLD (BALD): KL divergence between predictive distributions
- Parallel experimental design optimization
- Support for MCMC (accurate)
- Integration with Ax platform for Bayesian optimization
"""

import gc
import gc
import json
import argparse
from pathlib import Path
from typing import Optional
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
    "--acquisition",
    type=str,
    default="predictive",
    choices=["parametric", "predictive"],
    help="Acquisition function: 'parametric' for parametric KLD, 'predictive' for predictive KLD (BALD)",
)
parser.add_argument(
    "--ntestpoints",
    type=int,
    default=20,
    help="Number of test points for predictive KLD acquisition function",
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


def _single_predictive_kl_worker(args):
    x_old, y_old, x_new, x_test, idata_old, y_hyp, n_pred_samples, use_vi = args

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
        num_samples=n_pred_samples, num_warmup=1000, num_chains=2,
    )

    T_test = x_test[:, 0:1]
    p_test = x_test[:, 1:2]
    c2h_test = x_test[:, 2:3]
    c2m_test = x_test[:, 3:4]
    c2w_test = x_test[:, 4:5]
    c2i_test = x_test[:, 5:6]
    p_i_test = calculate_partial_pressures(p_test, c2h_test, c2m_test, c2w_test, c2i_test)

    pred_new = predict(idata_new, T_test, p_i_test, key="posterior", num_samples=n_pred_samples)
    pred_old = predict(idata_old, T_test, p_i_test, key="posterior", num_samples=n_pred_samples)

    n_test = x_test.shape[0]
    klds_i = np.zeros(n_test)
    for j in range(n_test):
        P_samples = pred_new[j, :].flatten()
        Q_samples = pred_old[j, :].flatten()
        klds_i[j] = kl_marginal(P_samples, Q_samples)

    return klds_i


def mp_predictive_KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    x_test: jnp.ndarray,
    idata_old: xarray.DataTree,
    n_hyp_samples: int = 5,
    n_pred_samples: int = 500,
    n_cores: int = 4,
    use_vi: bool = True,
):
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
    logger.debug(f"Generated {n_hyp_samples} hypothetical observations at candidate location")

    args_list = [
        (x_old, y_old, x_new, x_test, idata_old, jnp.array(y_new_samples[i].reshape(1, 1)), n_pred_samples, use_vi)
        for i in range(n_hyp_samples)
    ]

    n_procs = min(n_cores, n_hyp_samples)
    with NestablePool(n_procs) as pool:
        results = pool.map(_single_predictive_kl_worker, args_list)
    logger.debug("Completed predictive KL divergence computations in parallel")

    n_test = x_test.shape[0]
    klds = xarray.DataArray(
        dims=["hyp_sample", "test_point"],
        coords={"hyp_sample": np.arange(n_hyp_samples), "test_point": np.arange(n_test)},
        data=np.stack(results, axis=0),
    )
    return klds


def generate_test_grid(
    n_points: int = 20,
    T_range: tuple = (450.0, 600.0),
    p_range: tuple = (3.0, 9.0),
    c2h_range: tuple = (0.2, 4.0),
    c2m_range: tuple = (0.5, 1000.0),
    c2w_range: tuple = (0.5, 1000.0),
    c2i_range: tuple = (0.05, 1.0),
) -> jnp.ndarray:
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=n_points)

    lower_bounds = np.array([T_range[0], p_range[0], c2h_range[0], c2m_range[0], c2w_range[0], c2i_range[0]])
    upper_bounds = np.array([T_range[1], p_range[1], c2h_range[1], c2m_range[1], c2w_range[1], c2i_range[1]])

    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
    return jnp.array(scaled_sample)


def _single_parametric_kl_worker(args):
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
        var_names=["k0", "Ea_k", "A_oh", "dH_oh", "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r"],
    )
    return klds_i


def mp_parametric_KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: xarray.DataTree,
    n_hyp_samples: int = 5,
    n_cores: int = 4,
    use_vi: bool = True,
):
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
        results = pool.map(_single_parametric_kl_worker, args_list)
    logger.debug("Completed parametric KL divergence computations in parallel")

    klds = xarray.DataArray(
        dims=["hyp_sample", "parameter"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "parameter": ["k0", "Ea_k", "A_oh", "dH_oh", "A_h2", "dH_h2", "A_mix", "dH_mix", "sigma_r"],
        },
        data=np.stack(results, axis=0),
    )
    return klds


def mp_find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    proposals: jnp.ndarray,
    r_obs: jnp.ndarray,
    idata: xarray.DataTree,
    n_hyp_samples: int,
    n_workers: int,
    filename: str,
    use_vi: bool = True,
    acquisition: str = "parameter",
    x_test: Optional[jnp.ndarray] = None,
    n_pred_samples: int = 500,
):
    client = Client()
    parameters = [
        RangeParameterConfig(name="T", parameter_type="float", bounds=(450.0, 600.0)),
        RangeParameterConfig(name="p", parameter_type="float", bounds=(3.0, 9.0)),
        RangeParameterConfig(name="c2h", parameter_type="float", bounds=(0.2, 4.0)),
        RangeParameterConfig(name="c2m", parameter_type="float", bounds=(0.5, 1000.0)),
        RangeParameterConfig(name="c2w", parameter_type="float", bounds=(0.5, 1000.0)),
        RangeParameterConfig(name="c2i", parameter_type="float", bounds=(0.05, 1.0)),
    ]
    client.configure_experiment(parameters=parameters)
    client.configure_optimization(objective="KLD")
    for _ in range(n_iterations):
        trials = client.get_next_trials(max_trials=n_trials_per_iter)
        candidates = []
        indeces = []
        for trial_idx, parameters in trials.items():
            candidate = jnp.array([
                parameters["T"], parameters["p"], parameters["c2h"],
                parameters["c2m"], parameters["c2w"], parameters["c2i"],
            ])
            candidates.append(candidate)
            indeces.append(trial_idx)

        results = []
        for candidate in candidates:
            if acquisition == "parametric":
                KLD_vals = mp_parametric_KL_divergence(
                    x_old=proposals, y_old=r_obs, x_new=candidate,
                    idata_old=idata, n_hyp_samples=n_hyp_samples,
                    n_cores=n_workers, use_vi=use_vi,
                )
            elif acquisition == "predictive":
                if x_test is None:
                    raise ValueError("x_test must be provided for predictive acquisition")
                KLD_vals = mp_predictive_KL_divergence(
                    x_old=proposals, y_old=r_obs, x_new=candidate, x_test=x_test,
                    idata_old=idata, n_hyp_samples=n_hyp_samples,
                    n_pred_samples=n_pred_samples, n_cores=n_workers, use_vi=use_vi,
                )
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition}")
            results.append(KLD_vals)

        for trial_idx, KLD_vals in zip(indeces, results):
            mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
            std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()
            client.complete_trial(
                trial_index=trial_idx, raw_data={"KLD": (mean_KLD, std_KLD)},
            )
            logger.debug(f"Trial {trial_idx} has KLD {mean_KLD} +/- {std_KLD}")
    client.summarize()
    client.save_to_json_file(filename)
    best_parameters, expected_KLD, index, name = client.get_best_parameterization()
    return best_parameters, expected_KLD, index, name


def run_single_workflow(args, seed: int, exp_id: str, base_save_path: Path):
    seed_save_path = base_save_path / f"seed{seed:03d}"
    seed_save_path.mkdir(parents=True, exist_ok=True)
    
    num_workers = args.workers
    n_hyp_samples = args.KLsamples
    n_iterations = args.iterations
    n_opt_iterations = args.optiterations
    n_trials_per_iter = args.trialsperiteration
    acquisition = args.acquisition
    n_test_points = args.ntestpoints

    logger.info("=" * 70)
    logger.info(f"BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP (Seed: {seed})")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  KL samples: {n_hyp_samples}")
    logger.info(f"  Design iterations: {n_iterations}")
    logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Acquisition function: {acquisition}")
    if acquisition == "predictive":
        logger.info(f"  Test points: {n_test_points}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    x_test = None
    if acquisition == "predictive":
        logger.info("\n[SETUP] Generating test grid for predictive KLD...")
        x_test = generate_test_grid(n_points=n_test_points)
        logger.info(f"Generated {n_test_points} test points")

    logger.info("\n[STEP 0] Sampling from prior and finding initial experiment...")
    prior_idata = sample_prior(num_samples=5000)
    prior_idata.posterior = prior_idata.prior

    empty_proposals = jnp.empty((0, 6))
    empty_r_obs = jnp.empty((0, 1))

    best_parameters, expected_KLD, _, _ = mp_find_opt_candidate(
        n_iterations=n_opt_iterations,
        n_trials_per_iter=n_trials_per_iter,
        proposals=empty_proposals,
        r_obs=empty_r_obs,
        idata=prior_idata,
        n_hyp_samples=n_hyp_samples,
        n_workers=num_workers,
        filename=str(seed_save_path / "ax_client_000.json"),
        use_vi=True,
        acquisition=acquisition,
        x_test=x_test,
    )

    proposals = jnp.array([[
        best_parameters["T"], best_parameters["p"], best_parameters["c2h"],
        best_parameters["c2m"], best_parameters["c2w"], best_parameters["c2i"],
    ]])

    logger.info(f"Initial proposal: T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar")
    logger.info(f"Expected KLD: {expected_KLD['KLD'][0]:.4e}\n")

    trials_sum = []

    for i in range(1, n_iterations + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[ITERATION {i}/{n_iterations}] [Seed: {seed}]")
        logger.info(f"{'='*70}")

        T = proposals[:, 0:1]
        p = proposals[:, 1:2]
        c2h = proposals[:, 2:3]
        c2m = proposals[:, 3:4]
        c2w = proposals[:, 4:5]
        c2i = proposals[:, 5:6]

        logger.info(f"Conducting {proposals.shape[0]} experiments...")
        T, p_i, r_obs = generate_synthetic_data(
            T, p, c2h, c2m, c2w, c2i, seed=seed, rel_noise_std=0.1,
        )
        logger.info(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

        logger.info("Running variational inference...")
        idata = run_inference(
            T=T, p_i=p_i, r_obs=r_obs,
            num_samples=4000, num_warmup=4000, num_chains=8,
        )

        trace_filename = seed_save_path / f"trace_{i:03d}.nc"
        idata.to_netcdf(trace_filename)
        logger.info(f"Saved trace to: {trace_filename.name}")

        logger.info(f"Optimizing next experiment (KL divergence maximization)...")
        best_parameters, expected_KLD, _, _ = mp_find_opt_candidate(
            n_iterations=n_opt_iterations,
            n_trials_per_iter=n_trials_per_iter,
            proposals=proposals,
            r_obs=r_obs,
            idata=idata,
            n_hyp_samples=n_hyp_samples,
            n_workers=num_workers,
            filename=str(seed_save_path / f"ax_client_{i:03d}.json"),
            use_vi=True,
            acquisition=acquisition,
            x_test=x_test,
        )

        new_proposal = jnp.array([[
            best_parameters["T"], best_parameters["p"], best_parameters["c2h"],
            best_parameters["c2m"], best_parameters["c2w"], best_parameters["c2i"],
        ]])
        proposals = jnp.concatenate([proposals, new_proposal], axis=0)

        logger.info(f"Next optimal experiment:")
        logger.info(
            f"  T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar,"
            f"c2h={best_parameters['c2h']:.2f}, c2m={best_parameters['c2m']:.1f}, "
            f"c2w={best_parameters['c2w']:.1f}, c2i={best_parameters['c2i']:.2f}"
        )
        logger.info(f"  Expected KLD: {expected_KLD['KLD'][0]:.4e}")

        trial_summary = {
            "iteration": i,
            "seed": seed,
            "observed": {"x": proposals[:-1].tolist(), "y": r_obs.tolist()},
            "next_proposal": {
                "x": new_proposal.tolist()[0],
                "expected_KLD": float(expected_KLD["KLD"][0]),
                "KLD_std": float(expected_KLD["KLD"][1]) if len(expected_KLD["KLD"]) > 1 else None,
            },
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
    logger.info(f"SEQUENTIAL DESIGN COMPLETE (Seed: {seed})")
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
    base_seed = args.seed
    n_seeds = args.n_seeds
    
    global EXP_ID
    EXP_ID = namer.generate()
    
    base_save_path = Path("/lustre/work/ws/ws1/ul_qmj36-BIBO_workspace")  / "data" / f"{EXP_ID}"
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

        # Drop large per-seed references before the next seed starts.
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
