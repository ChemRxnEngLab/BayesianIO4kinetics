"""
Bayesian Optimal Experimental Design using KL Divergence (Pyro Version)

This module just runs a single dummy run to demonstrate the parametric uncertainty for the talk.
"""

import functools
import json
import argparse
from pathlib import Path
import arviz as az
import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import xarray
import namer
from ax import Client, RangeParameterConfig
from NestPools import NestablePool

from HSA_Mo_B_004_pyro import (
    run_variational_inference,
    run_inference,
    predict,
    sample_prior,
    generate_synthetic_data,
    calculate_partial_pressures,
)


def main():
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

    # Find first experiment from prior (no observations yet)
    proposals = torch.tensor(
        [
            [525, 2.0, 0.5, 10, 0.5],
            [550, 2.0, 0.5, 10, 0.5],
            [575, 2.0, 0.5, 10, 0.5],
            [525, 15.0, 0.5, 10, 0.5],
            [550, 15.0, 0.5, 10, 0.5],
            [575, 15.0, 0.5, 10, 0.5],
            [525, 10.0, 2.0, 10, 0.5],
            [550, 10.0, 2.0, 10, 0.5],
            [575, 10.0, 2.0, 10, 0.5],
        ]
    )

    # Extract experimental conditions
    T = proposals[:, 0:1]
    p = proposals[:, 1:2]
    c2h = proposals[:, 2:3]
    f = proposals[:, 3:4]
    c2i = proposals[:, 4:5]

    # Conduct experiments (generate synthetic observations)
    print(f"Conducting {proposals.shape[0]} experiments...")
    T, p_i, r_obs = generate_synthetic_data(T, p, c2h, f, c2i, seed=42)
    print(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

    # Run Bayesian inference (Variational Inference for speed)
    print("Running variational inference...")
    idata = run_variational_inference(
        T=T,
        p_i=p_i,
        r_obs=r_obs.squeeze(),
        num_steps=10000,
        learning_rate=0.01,
        num_samples=4000,
    )

    # Add prior samples for comparison
    print("Adding prior samples to inference data...")
    prior_samples = sample_prior(num_samples=4000)
    idata.add_groups(prior=prior_samples.prior)

    # Save inference results
    trace_filename = save_path / f"trace.nc"
    idata.to_netcdf(trace_filename)
    print(f"Saved trace to: {trace_filename.name}")


if __name__ == "__main__":
    save_path = Path.cwd() / "data" / "C_001"
    save_path.mkdir(parents=True)
    main()
