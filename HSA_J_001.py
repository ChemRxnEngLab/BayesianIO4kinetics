import argparse
import datetime
import gc
import logging

import itertools
import json
from pathlib import Path
from typing import Callable, Optional, Iterator
import arviz as az
import numpy as np
from scipy.stats import qmc
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import xarray
import namer
from ax import Client, RangeParameterConfig
from NestPools import NestablePool
import numpyro

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

numpyro.set_host_device_count(48)
import jax
import jax.numpy as jnp

from HSA_GTMo_C_004_numpyro import (
    generate_synthetic_data,
    calculate_partial_pressures,
    TRUE_PARAMS,
)

from HSA_Mo_D_001_numpyro import run_inference as run_inference_Mo1
from HSA_Mo_D_001_numpyro import predict as predict_Mo1
from HSA_Mo_D_002_numpyro import run_inference as run_inference_Mo2
from HSA_Mo_D_002_numpyro import predict as predict_Mo2
from HSA_Mo_D_003_numpyro import run_inference as run_inference_Mo3
from HSA_Mo_D_003_numpyro import predict as predict_Mo3
from HSA_Mo_D_004_numpyro import run_inference as run_inference_Mo4
from HSA_Mo_D_004_numpyro import predict as predict_Mo4


parser = argparse.ArgumentParser()
parser.add_argument(
    "--iterations",
    type=int,
    default=8,
    help="Number of points to additionally sample in the experiment",
)
parser.add_argument(
    "--optiterations",
    type=int,
    # default=2,
    default=15,
    help="Number of optimization iterations for Ax",
)
parser.add_argument(
    "--trialsperiteration",
    type=int,
    default=1,
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
    default=5,
    help="Number of different seeds to run (for ensemble studies)",
)

models = ["Mo1", "Mo2", "Mo3", "Mo4"]

predict_funs: dict[str, Callable] = {
    "Mo1": predict_Mo1,
    "Mo2": predict_Mo2,
    "Mo3": predict_Mo3,
    "Mo4": predict_Mo4,
}

inference_funs: dict[str, Callable] = {
    "Mo1": run_inference_Mo1,
    "Mo2": run_inference_Mo2,
    "Mo3": run_inference_Mo3,
    "Mo4": run_inference_Mo4,
}

PARAM_RANGES = {
    "T": (450.0, 600.0),
    "p": (3.0, 9.0),
    "c2h": (0.2, 4.0),
    "c2m": (0.5, 1000.0),
    "c2w": (0.5, 1000.0),
    "c2i": (0.05, 1.0),
}


def JSD_marginal(
    p_samples: jnp.ndarray,
    q_samples: jnp.ndarray,
    eps=1e-12,
) -> float:
    """Calculate Jensen-Shannon Divergence between two sets of predictions.

    Args:
        r_pred1: First set of predicted reaction rates as a JAX array.
        r_pred2: Second set of predicted reaction rates as a JAX array.
        num_bins: Number of bins to use for histogram estimation.
    Returns:
        JSD values for each condition as a JAX array.
    """
    kde_P = gaussian_kde(p_samples)
    kde_Q = gaussian_kde(q_samples)

    grid = np.linspace(
        min(p_samples.min(), q_samples.min()),
        max(p_samples.max(), q_samples.max()),
        1000,
    )

    p_vals = kde_P(grid) + eps
    q_vals = kde_Q(grid) + eps

    m_vals = 1 / 2 * (p_vals + q_vals)  #

    kld_pm = simpson(p_vals * np.log(p_vals / m_vals), grid)
    kld_qm = simpson(q_vals * np.log(q_vals / m_vals), grid)

    jsd = 1 / 2 * (kld_pm + kld_qm)
    return jsd


def JSD_between_predictions(
    r_pred1: jnp.ndarray,
    r_pred2: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate Jensen-Shannon Divergence between two sets of predictions.

    Args:
        r_pred1: First set of predicted reaction rates as a JAX array.
        r_pred2: Second set of predicted reaction rates as a JAX array.
    Returns:
        JSD values for each condition as a JAX array.
    """
    # shape of r_pred: (10, 1, 1, num_samples)
    num_times = r_pred1.shape[0]

    jsds = jnp.empty((0,))
    for i in range(num_times):
        p_samples = jnp.array(r_pred1[i, 0, 0, :].squeeze())
        q_samples = jnp.array(r_pred2[i, 0, 0, :].squeeze())
        jsd = JSD_marginal(p_samples, q_samples)
        jsds = jnp.append(jsds, jsd)
    return jsds


def calculate_disagreement(
    idatas: list[az.InferenceData],
    candidate: jnp.ndarray,
    num_samples: int = 1000,
) -> jnp.ndarray:
    """Calculate disagreement metric for a candidate parameter set.

    Args:
        idatas: List of InferenceData objects from different models.
        candidate: Candidate parameter set as a JAX array.
    Returns:
        Disagreement metric as a float.
    """
    # extract conditions and calculate partial pressures
    T_cand = candidate[0:1].reshape(-1, 1)
    p_cand = candidate[1:2].reshape(-1, 1)
    c2h_cand = candidate[2:3].reshape(-1, 1)
    c2m_cand = candidate[3:4].reshape(-1, 1)
    c2w_cand = candidate[4:5].reshape(-1, 1)
    c2i_cand = candidate[5:6].reshape(-1, 1)

    pi_cand = calculate_partial_pressures(
        p_cand,
        c2h_cand,
        c2m_cand,
        c2w_cand,
        c2i_cand,
    )

    r_preds:list[jnp.ndarray] = []
    for Mo in models:
        r_pred: jnp.ndarray = predict_funs[Mo](
            idata=idatas[models.index(Mo)],
            T=T_cand,
            p_i=pi_cand,
            num_samples=num_samples,
            w_noise=True,
        )
        r_preds.append(r_pred)

    # calculate the pairwise JSDs and add them up
    pairwise_models: Iterator[tuple[jnp.ndarray, ...]] = itertools.combinations(r_preds, 2)
    pairwise_disagreements:list[jnp.ndarray] = []
    for r_pred1, r_pred2 in pairwise_models:
        pairwise_disagreements_over_t = JSD_between_predictions(
            r_pred1, r_pred2
        )  # shape (10,)
        pairwise_disagreement = jnp.mean(pairwise_disagreements_over_t)
        pairwise_disagreements.append(pairwise_disagreement)
    disagreement = jnp.sum(jnp.array(pairwise_disagreements))
    return disagreement


def find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    idatas: list[az.InferenceData],
    filename: str,
):
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
            bounds=(0.2, 4.0),
        ),
        RangeParameterConfig(
            name="c2m",
            parameter_type="float",
            bounds=(0.5, 1000.0),
        ),
        RangeParameterConfig(
            name="c2w",
            parameter_type="float",
            bounds=(0.5, 1000.0),
        ),
        RangeParameterConfig(
            name="c2i",
            parameter_type="float",
            bounds=(0.05, 1.0),
        ),
    ]
    client.configure_experiment(
        parameters=parameters,
    )
    client.configure_optimization(objective="disagreement")
    for _ in range(n_iterations):
        trials = client.get_next_trials(max_trials=n_trials_per_iter)
        for trial_idx, trial_parameters in trials.items():
            candidate = jnp.array(
                [
                    trial_parameters["T"], #type: ignore
                    trial_parameters["p"], #type: ignore
                    trial_parameters["c2h"], #type: ignore
                    trial_parameters["c2m"], #type: ignore
                    trial_parameters["c2w"], #type: ignore
                    trial_parameters["c2i"], #type: ignore
                ],
            )
            disagreement = calculate_disagreement(idatas, candidate)
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"disagreement": disagreement.item()},
            )
            logger.debug(f"Trial {trial_idx} has disagreement {disagreement.item()}")
    client.summarize()
    client.save_to_json_file(filename)
    best_parameters, objective, index, name = client.get_best_parameterization()
    return best_parameters, objective, index, name


def LHSDOE(n_obs: int, seed: Optional[int] = None):
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    sample = sampler.random(n_obs)

    # Scale samples to the appropriate ranges
    T_lhs = qmc.scale(sample[:, 0:1], PARAM_RANGES["T"][0], PARAM_RANGES["T"][1])
    p_lhs = qmc.scale(sample[:, 1:2], PARAM_RANGES["p"][0], PARAM_RANGES["p"][1])
    c2h_lhs = qmc.scale(sample[:, 2:3], PARAM_RANGES["c2h"][0], PARAM_RANGES["c2h"][1])
    c2m_lhs = qmc.scale(sample[:, 3:4], PARAM_RANGES["c2m"][0], PARAM_RANGES["c2m"][1])
    c2w_lhs = qmc.scale(sample[:, 4:5], PARAM_RANGES["c2w"][0], PARAM_RANGES["c2w"][1])
    c2i_lhs = qmc.scale(sample[:, 5:6], PARAM_RANGES["c2i"][0], PARAM_RANGES["c2i"][1])

    proposals = jnp.column_stack([T_lhs, p_lhs, c2h_lhs, c2m_lhs, c2w_lhs, c2i_lhs])
    return proposals

def _single_runner(args):
    Mo, T_obs, p_obs,r_obs = args
    idata = inference_funs[Mo](
        T_obs,
        p_obs,
        r_obs,
        num_warmup=4000,
        num_samples=4000,
        num_chains=8,
    )
    return idata


def run_single_workflow(args, seed: int, exp_id: str, base_save_path: Path):
    """Run the complete model discrimination workflow with a specific seed.

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

    n_iterations = args.iterations
    n_opt_iterations = args.optiterations
    n_trials_per_iter = args.trialsperiteration

    logger.info("=" * 70)
    logger.info(f"BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP (Seed: {seed})")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Design iterations: {n_iterations}")
    logger.info(f"  Optimization iterations: {n_opt_iterations}")
    logger.info(f"  Experiment ID: {exp_id}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)

    # Conduct 8 parameterization runs and then model discrimination runs
    initial_proposals = LHSDOE(12, seed=seed)
    T = initial_proposals[:, 0:1]
    p = initial_proposals[:, 1:2]
    c2h = initial_proposals[:, 2:3]
    c2m = initial_proposals[:, 3:4]
    c2w = initial_proposals[:, 4:5]
    c2i = initial_proposals[:, 5:6]
    # generatte synthetic observations
    T_obs, p_obs, r_obs = generate_synthetic_data(
        T,
        p,
        c2h,
        c2m,
        c2w,
        c2i,
        seed=seed,
        rel_noise_std=0.1,
    )

    # Run all specified models in parallel
    argslist = [(Mo, T_obs, p_obs, r_obs) for Mo in models]
    with NestablePool(4) as pool:
        idatas = pool.map(
            _single_runner,
            argslist,
        )
    for Mo, idata in zip(models, idatas):
        trace_filename = seed_save_path / f"{Mo}_trace_000.nc"
        idata.to_netcdf(trace_filename)  

    logger.info("Initial Training complete")
    trials_sum = []
    proposals = initial_proposals

    for i in range(1, n_iterations + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[ITERATION {i}/{n_iterations}] [Seed: {seed}]")
        logger.info(f"{'='*70}")
        # determine the next best point
        best_parameters, metric, _, _ = find_opt_candidate(
            n_iterations=n_opt_iterations,
            n_trials_per_iter=n_trials_per_iter,
            idatas=idatas,
            filename=str(seed_save_path / f"ax_client_{i:03d}.json"),
        )
        next_proposal = jnp.array(
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
        logger.info(f"Next optimal experiment:")
        logger.info(
            f"  T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar,"
            f"c2h={best_parameters['c2h']:.2f}, c2m={best_parameters['c2m']:.1f}, "
            f"c2w={best_parameters['c2w']:.1f}, c2i={best_parameters['c2i']:.2f}"
        )
        logger.info(f"  Expected disagreement: {metric['disagreement'][0]:.4e}")
        # add the proposal and generate the data
        proposals = jnp.concatenate([proposals, next_proposal], axis=0)
        T = proposals[:, 0:1]
        p = proposals[:, 1:2]
        c2h = proposals[:, 2:3]
        c2m = proposals[:, 3:4]
        c2w = proposals[:, 4:5]
        c2i = proposals[:, 5:6]
        T_obs, p_obs, r_obs = generate_synthetic_data(
            T,
            p,
            c2h,
            c2m,
            c2w,
            c2i,
            seed=seed,
            rel_noise_std=0.1,
        )

        # Run all specified models in parallel
        argslist = [(Mo, T_obs, p_obs, r_obs) for Mo in models]
        with NestablePool(4) as pool:
            idatas = pool.map(
                _single_runner,
                argslist,
            )
        for Mo, idata in zip(models, idatas):
            trace_filename = seed_save_path / f"{Mo}_trace_{i:03d}.nc"
            idata.to_netcdf(trace_filename)    
        # Store iteration summary
        trial_summary = {
            "iteration": i,
            "seed": seed,
            "observed": {
                "x": proposals.tolist(),  # All proposals
                "y": r_obs.tolist(),
            },
            "next_proposal": {
                "x": next_proposal.tolist()[0],
                "disagreement": metric["disagreement"][0],
            },
            "trace_path": str(trace_filename),
        }
        trials_sum.append(trial_summary)

        # Update and save experiment summary
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
    logger.info(f"MODEL DISCRIMINATION COMPLETE (Seed: {seed})")
    logger.info(f"{'='*70}")
    logger.info(f"Total experiments conducted: {proposals.shape[0]}")
    logger.info(f"Results saved to: {seed_save_path}")
    logger.info(f"{'='*70}\n")

    # Drop large per-seed references before the next seed starts.
    del idata, proposals, r_obs, trials_sum
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
    import datetime
    args = parser.parse_args()
    main(args)
