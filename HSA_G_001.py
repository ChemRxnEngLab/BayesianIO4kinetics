"""
Koshany experimental plan (Numpyro Version)

This module implements the original Koschany et al. experimental design.

"""

import json
import argparse
from pathlib import Path
import arviz as az
import numpy as np

from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import xarray
import namer
import numpyro

numpyro.set_platform("cpu")
numpyro.set_host_device_count(48)
import jax
import jax.numpy as jnp

from NestPools import NestablePool


from HSA_Mo_B_012 import (
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
    default=32,  # TODO
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


def main(args):
    """
    Run sequential Bayesian optimal experimental design loop.

    Workflow:
    1. Sample from prior and find initial optimal experiment
    2. For each iteration:
       - Conduct experiments at current proposals
       - Run Bayesian inference (VI)
       - Find next optimal experiment via KL divergence maximization
       - Save results and update experiment summary

    Args:
        args: Command-line arguments with workers, KLsamples, iterations, optiterations
    """
    # Extract configuration from args
    num_workers = args.workers
    n_hyp_samples = args.KLsamples
    n_iterations = args.iterations
    n_opt_iterations = args.optiterations
    n_trials_per_iter = args.trialsperiteration

    print("=" * 70)
    print("KOSCHANY DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  KL samples: {n_hyp_samples}")
    print(f"  Design iterations: {n_iterations}")
    print(f"  Optimization iterations: {n_opt_iterations}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    all_proposals = koschany_design()
    n_iterations = all_proposals.shape[0]

    # Step 1: Sample prior and find initial optimal experiment
    prior_idata = sample_prior(num_samples=4000)
    prior_idata.posterior   = prior_idata.prior.copy()

    # Find first experiment from prior (no observations yet)
    proposals = jnp.empty((0, 6))
    r_obs = jnp.empty((0, 1))

    # Step 2: Full factorial design loop
    trials_sum = []

    for i in range(0, n_iterations):
        print(f"\n{'='*70}")
        print(f"[ITERATION {i}/{n_iterations}]")
        print(f"{'='*70}")

        new_proposal = all_proposals[i : i + 1, :]

        # append the new proposal we just calculated KLD for and condition the model
        proposals = jnp.concatenate([proposals, new_proposal], axis=0)

        # Extract experimental conditions
        T = proposals[:, 0:1]
        p = proposals[:, 1:2]
        c2h = proposals[:, 2:3]
        c2m = proposals[:, 3:4]
        c2w = proposals[:, 4:5]
        c2i = proposals[:, 5:6]

        # Conduct experiments (generate synthetic observations)
        print(f"Conducting {proposals.shape[0]} experiments...")
        T, p_i, r_obs = generate_synthetic_data(
            T, p, c2h, c2m, c2w, c2i, seed=42, rel_noise_std=0.1
        )
        print(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

        print("Running MCMC inference...")
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs,
            num_samples=4000,
            num_warmup=4000,
            num_chains=8,
        )
        # assign the newly trained inference data to prior for next round
        prior_idata = idata.copy()

        # Save inference results
        trace_filename = save_path / f"trace_{i:03d}.nc"
        idata.to_netcdf(trace_filename)
        print(f"Saved trace to: {trace_filename.name}")

        # Store iteration summary
        trial_summary = {
            "iteration": i,
            "observed": {
                "x": proposals.tolist(),
                "y": r_obs.tolist(),
            },
            "predicted_before_obs": {
                "expected_KLD": None,
                "std_KLD": None,
            },
            "trace_path": str(trace_filename),
        }
        trials_sum.append(trial_summary)

        # Update and save experiment summary
        exp_summary["trials"] = trials_sum
        summary_file = save_path / f"experiment_summary_{EXP_ID}.json"
        with open(summary_file, "w") as f:
            json.dump(exp_summary, f, indent=2)

        print(f"Updated experiment summary: {summary_file.name}")

    # Final summary
    print(f"\n{'='*70}")
    print("FF DESIGN COMPLETE")
    print(f"{'='*70}")
    print(f"Total experiments conducted: {proposals.shape[0]}")
    print(f"Results saved to: {save_path}")
    print(f"Experiment ID: {EXP_ID}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    global EXP_ID, exp_summary, save_path
    EXP_ID = namer.generate()
    exp_summary = {
        "experiment_id": EXP_ID,
        "root file": str(Path(__file__)),
        "args": vars(args),
    }
    save_path = Path.cwd() / "data" / f"{EXP_ID}"
    save_path.mkdir(parents=True)
    main(args)
    with open(save_path / f"experiment_summary_{EXP_ID}.json", "w") as f:
        f.write(json.dumps(exp_summary))
