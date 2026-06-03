"""
LHS experimental plan

This module implements a LHS experimental design as per Reviewer 2.

"""

import json
import argparse
from pathlib import Path
import numpy as np

from scipy.stats import qmc
from scipy.integrate import simpson
import namer
import numpyro

numpyro.set_platform("cpu")
numpyro.set_host_device_count(12)
import jax
import jax.numpy as jnp

print(f"Jax devices: {jax.devices()}")

from HSA_Mo_B_011_numpyro import (
    run_inference,
    predict,
    sample_prior,
    generate_synthetic_data,
    calculate_partial_pressures,
)

parser = argparse.ArgumentParser()
#All of these are unneccessary for the LHS design, but we keep them here for consistency with the other experiments and potential future use when we implement the KL-based design
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

def LHS_design():
    # Define the ranges
    T_range = (450.0, 600.0)
    p_range = (3.0, 9.0)
    c2h_range = (0.2, 4.0)
    c2m_range = (0.5, 1000.0)  # Logarithmic
    c2w_range = (0.5, 1000.0)  # Logarithmic
    c2i_range = (0.05, 1.0)

    n_samples = 64
    n_params = 6  # Number of parameters

    # Generate LHS samples in [0, 1]
    sampler = qmc.LatinHypercube(d=n_params)
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

    print("=" * 70)
    print("LHS DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    # print(f"  Workers: {num_workers}")
    # print(f"  KL samples: {n_hyp_samples}")
    # print(f"  Design iterations: {n_iterations}")
    # print(f"  Optimization iterations: {n_opt_iterations}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    all_proposals = LHS_design()
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
    print("LHS DESIGN COMPLETE")
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
