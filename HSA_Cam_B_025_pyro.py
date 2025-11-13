"""
Bayesian Optimal Experimental Design using KL Divergence (Pyro Version)

This module implements Bayesian optimal experimental design for kinetic parameter
estimation using KL divergence as the utility function. Refactored to work with
Pyro probabilistic programming framework (PyTorch backend).

Key Features:
- KL divergence computation between posterior distributions
- Sequential and parallel experimental design optimization
- Support for both Variational Inference (fast) and MCMC (accurate)
- Integration with Ax platform for Bayesian optimization

Main Functions:
- kl_marginal: Compute KL divergence between marginal distributions using KDE
- kl_divergence_between_models: Compute KL divergence for all parameters
- KL_divergence: Sequential KL computation for single candidate (with optional VI)
- mp_KL_divergence: Parallel KL computation using multiprocessing
- sp_find_opt_candidate: Sequential Bayesian optimization of experimental design
- mp_find_opt_candidate: Parallel Bayesian optimization (faster)

Design Philosophy:
- Use VI (variational inference) for rapid exploration
- Use MCMC for final accurate inference
- Parallelize across hypothetical observations for speedup
- Use Ax platform for adaptive experimental design
"""

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


from HSA_Mo_B_009_pyro import (
    run_variational_inference,
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
    default=16,  # TODO
    help="Number of points to additionally sample in the experiment",
)
parser.add_argument(
    "--optiterations",
    type=int,
    default=8,
    help="Number of optimization iterations for Ax",
)
parser.add_argument(
    "--trialsperiteration",
    type=int,
    default=3,
    help="Number of trials per optimization iteration for Ax",
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
    # Compute KL divergence using marginal distributions
    for var in old_samples.data_vars:
        P_samples = new_samples[var].values.flatten()
        Q_samples = old_samples[var].values.flatten()
        kl_div = kl_marginal(P_samples, Q_samples)
        klds.append(kl_div)
    return np.array(klds)


def KL_divergence(
    x_old: torch.Tensor,
    y_old: torch.Tensor,
    x_new: torch.Tensor,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    use_vi: bool = True,
):
    """
    Compute KL divergence between posterior distributions before and after adding hypothetical data.

    KL(P_new || P_old) = E_P_new[log P_new(θ) - log P_old(θ)]

    Args:
        x_old: Current experimental conditions (n_current, 5) [T, p, c2h, f, c2i]
        y_old: Current observations (n_current, 1)
        x_new: New experimental condition to test (5,) [T, p, c2h, f, c2i]
        idata_old: Current posterior samples (ArviZ InferenceData)
        n_hyp_samples: Number of hypothetical observations to average over
        use_vi: If True, use variational inference (faster); if False, use MCMC (slower but more accurate)

    Returns:
        xarray.DataArray of KL divergences with dimensions (hyp_sample, parameter)
    """
    # Extract conditions from x_new
    T_new = x_new[0:1].reshape(-1, 1)
    p_new = x_new[1:2].reshape(-1, 1)
    c2h_new = x_new[2:3].reshape(-1, 1)
    f_new = x_new[3:4].reshape(-1, 1)
    c2i_new = x_new[4:5].reshape(-1, 1)

    # Calculate partial pressures for new condition
    p_i_new = calculate_partial_pressures(p_new, c2h_new, f_new, c2i_new)

    # Step 1: Generate hypothetical observations at the new experimental condition
    y_new_samples = predict(
        idata_old,
        T_new,
        p_i_new,
        key="posterior",
        num_samples=n_hyp_samples,
    ).reshape(-1, 1)
    print(f"Generated {n_hyp_samples} hypothetical observations at new condition")

    # Step 2: Compute KL divergence for each hypothetical observation
    klds = xarray.DataArray(
        dims=["hyp_sample", "parameter"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "parameter": [
                "k0",
                "Ea_k",
                "A_oh",
                "dH_oh",
                "A_h2",
                "dH_h2",
                "A_mix",
                "dH_mix",
                "sigma_r",
            ],
        },
        data=np.zeros((n_hyp_samples, 9)),
    )

    for i in range(n_hyp_samples):
        print(f"Processing hypothetical sample {i+1}/{n_hyp_samples}")

        # Get the i-th hypothetical observation
        y_hyp = torch.tensor(y_new_samples[i].reshape(1, 1), dtype=torch.float32)

        # Create augmented dataset: combine current data with hypothetical observation
        x_augmented = torch.vstack([x_old, x_new.reshape(1, -1)])
        y_augmented = torch.vstack([y_old, y_hyp])

        # Extract conditions and generate synthetic data structure
        T_aug = x_augmented[:, 0:1]
        p_aug = x_augmented[:, 1:2]
        c2h_aug = x_augmented[:, 2:3]
        f_aug = x_augmented[:, 3:4]
        c2i_aug = x_augmented[:, 4:5]

        # Calculate partial pressures for augmented data
        p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, f_aug, c2i_aug)

        # Sample from the new posterior (with augmented data)
        if use_vi:
            # Fast variational inference
            idata_new = run_variational_inference(
                T_aug,
                p_i_aug,
                y_augmented.squeeze(),
                num_steps=5000,
                num_draws=2000,
            )
        else:
            # Accurate MCMC inference
            idata_new = run_inference(
                T_aug,
                p_i_aug,
                y_augmented.squeeze(),
                num_samples=2000,
                num_warmup=1000,
                num_chains=2,
            )

        # Step 3: Compute KL divergence between posteriors
        klds_i = kl_divergence_between_models(
            idata_new,
            idata_old,
            key_new="posterior",
            key_old="posterior",
            n_samples=1000,
            var_names=[
                "k0",
                "Ea_k",
                "A_oh",
                "dH_oh",
                "A_h2",
                "dH_h2",
                "A_mix",
                "dH_mix",
                "sigma_r",
            ],
        )
        klds.loc[i, :] = klds_i

    # Step 4: Return KL divergences across all hypothetical samples
    return klds


def _single_kl_worker(args):
    """
    Worker function for computing KL divergence for a single hypothetical observation.
    Must be at top level for multiprocessing on Windows.
    """
    x_old, y_old, x_new, idata_old, y_hyp, use_vi = args

    # Create augmented dataset
    x_augmented = torch.vstack([x_old, x_new.reshape(1, -1)])
    y_augmented = torch.vstack([y_old, y_hyp])

    # Extract conditions
    T_aug = x_augmented[:, 0:1]
    p_aug = x_augmented[:, 1:2]
    c2h_aug = x_augmented[:, 2:3]
    f_aug = x_augmented[:, 3:4]
    c2i_aug = x_augmented[:, 4:5]

    # Calculate partial pressures
    p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, f_aug, c2i_aug)

    # Run inference on augmented data
    if use_vi:
        # Fast variational inference
        idata_new = run_variational_inference(
            T_aug,
            p_i_aug,
            y_augmented.squeeze(),
            num_steps=10000,
            num_draws=4000,
        )
    else:
        # Accurate MCMC inference
        idata_new = run_inference(
            T_aug,
            p_i_aug,
            y_augmented.squeeze(),
            num_samples=2000,
            num_warmup=1000,
            num_chains=2,
        )

    # Compute KL divergence
    klds_i = kl_divergence_between_models(
        idata_new,
        idata_old,
        key_new="posterior",
        key_old="posterior",
        n_samples=1000,
        var_names=[
            "k0",
            "Ea_k",
            "A_oh",
            "dH_oh",
            "A_h2",
            "dH_h2",
            "A_mix",
            "dH_mix",
            "sigma_r",
        ],
    )
    return klds_i


def _candidate_kl_worker(
    args: list[
        torch.Tensor, torch.Tensor, torch.Tensor, az.InferenceData, int, int, bool
    ],
):
    """Top-level worker for parallelizing over candidate proposals."""
    x_old, y_old, x_new, idata_old, n_hyp_samples, n_cores, use_vi = args
    return mp_KL_divergence(
        x_old=x_old,
        y_old=y_old,
        x_new=x_new.squeeze(),
        idata_old=idata_old,
        n_hyp_samples=n_hyp_samples,
        n_cores=n_cores,
        use_vi=use_vi,
    )


def mp_KL_divergence(
    x_old: torch.Tensor,
    y_old: torch.Tensor,
    x_new: torch.Tensor,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    n_cores: int = 4,
    use_vi: bool = True,
):
    """
    Compute KL divergence in parallel across multiple hypothetical observations.

    Args:
        x_old: Current experimental conditions (n_current, 5)
        y_old: Current observations (n_current, 1)
        x_new: New experimental condition to test (5,)
        idata_old: Current posterior samples
        n_hyp_samples: Number of hypothetical observations
        n_cores: Number of parallel workers
        use_vi: If True, use VI (fast); if False, use MCMC (accurate)

    Returns:
        xarray.DataArray of KL divergences
    """
    # Extract conditions from x_new
    T_new = x_new[0:1].reshape(-1, 1)
    p_new = x_new[1:2].reshape(-1, 1)
    c2h_new = x_new[2:3].reshape(-1, 1)
    f_new = x_new[3:4].reshape(-1, 1)
    c2i_new = x_new[4:5].reshape(-1, 1)

    # Calculate partial pressures
    p_i_new = calculate_partial_pressures(p_new, c2h_new, f_new, c2i_new)

    # Step 1: Generate hypothetical observations at the new experimental condition
    y_new_samples = predict(
        idata_old,
        T_new,
        p_i_new,
        key="posterior",
        num_samples=n_hyp_samples,
    ).reshape(-1, 1)
    print(f"Generated {n_hyp_samples} hypothetical observations at new condition")

    # Step 2: Compute KL divergence for each hypothetical observation in parallel
    args_list = [
        (
            x_old,
            y_old,
            x_new,
            idata_old,
            torch.tensor(y_new_samples[i].reshape(1, 1), dtype=torch.float32),
            use_vi,
        )
        for i in range(n_hyp_samples)
    ]
    n_procs = min(n_cores, n_hyp_samples)
    with NestablePool(n_procs) as pool:
        results = pool.map(_single_kl_worker, args_list)
    print("Completed KL divergence computations in parallel")

    klds = xarray.DataArray(
        dims=["hyp_sample", "parameter"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "parameter": [
                "k0",
                "Ea_k",
                "A_oh",
                "dH_oh",
                "A_h2",
                "dH_h2",
                "A_mix",
                "dH_mix",
                "sigma_r",
            ],
        },
        data=np.stack(results, axis=0),
    )

    # Step 4: Return KL divergences across all hypothetical samples
    return klds


def FFD():
    # Define the parameter ranges for the full factorial design
    T_range = np.array([500, 600])
    p_range = np.array([2, 20])
    c2h_range = np.array([0.5, 2])
    f_range = np.array([1, 1000])
    c2i_range = np.array([0.1, 1])

    # Create the full factorial design grid
    Tgrid, pgrid, c2hgrid, fgrid, c2igrid = np.meshgrid(
        T_range, p_range, c2h_range, f_range, c2i_range
    )

    # Combine the grids into a list of proposals
    proposals = np.vstack(
        [Tgrid.ravel(), pgrid.ravel(), c2hgrid.ravel(), fgrid.ravel(), c2igrid.ravel()]
    ).T

    # reorder for better space-filling
    proposals_new = proposals[
        [
            2,
            10,
            18,
            26,
            6,
            14,
            22,
            30,
            3,
            11,
            19,
            27,
            1,
            9,
            17,
            25,
            0,
            8,
            16,
            24,
            4,
            12,
            20,
            28,
            5,
            13,
            21,
            29,
            7,
            15,
            23,
            31,
        ],
        :,
    ]
    return torch.tensor(proposals_new)


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
    print("FULL FACTORIAL DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  KL samples: {n_hyp_samples}")
    print(f"  Design iterations: {n_iterations}")
    print(f"  Optimization iterations: {n_opt_iterations}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    all_proposals = FFD()
    n_iterations = all_proposals.shape[0]

    # Step 1: Sample prior and find initial optimal experiment
    prior_idata = sample_prior(num_samples=4000)
    prior_idata.add_groups(posterior=prior_idata.prior)

    # Find first experiment from prior (no observations yet)
    proposals = torch.empty((0, 5), dtype=torch.float32)
    r_obs = torch.empty((0, 1), dtype=torch.float32)

    # Step 2: Full factorial design loop
    trials_sum = []

    for i in range(0, n_iterations):
        print(f"\n{'='*70}")
        print(f"[ITERATION {i}/{n_iterations}]")
        print(f"{'='*70}")

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

        # append the new proposal we just calculated KLD for and condition the model
        proposals = torch.cat([proposals, new_proposal], dim=0)

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
            num_draws=4000,
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
                "x": proposals.tolist(),  # All proposals except the new one
                "y": r_obs.tolist(),
            },
            "predicted_before_obs": {
                "expected_KLD": mean_KLD,
                "std_KLD": std_KLD,
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
