"""
Bayesian Optimal Experimental Design using KL Divergence (Numpyro Version)

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

from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import xarray
import namer
from ax import Client, RangeParameterConfig
import numpyro

numpyro.set_host_device_count(48)
import jax
import jax.numpy as jnp

from NestPools import NestablePool

# informative priors
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
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: xarray.DataTree,
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
    c2m_new = x_new[3:4].reshape(-1, 1)
    c2w_new = x_new[4:5].reshape(-1, 1)
    c2i_new = x_new[5:6].reshape(-1, 1)

    # Calculate partial pressures for new condition
    p_i_new = calculate_partial_pressures(p_new, c2h_new, c2m_new, c2w_new, c2i_new)

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
        y_hyp = jnp.array(y_new_samples[i].reshape(1, 1))

        # Create augmented dataset: combine current data with hypothetical observation
        x_augmented = jnp.vstack([x_old, x_new.reshape(1, -1)])
        y_augmented = jnp.vstack([y_old, y_hyp])

        # Extract conditions and generate synthetic data structure
        T_aug = x_augmented[:, 0:1]
        p_aug = x_augmented[:, 1:2]
        c2h_aug = x_augmented[:, 2:3]
        c2m_aug = x_augmented[:, 3:4]
        c2w_aug = x_augmented[:, 4:5]
        c2i_aug = x_augmented[:, 5:6]

        # Calculate partial pressures for augmented data
        p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, c2m_aug, c2w_aug, c2i_aug)

        # Sample from the new posterior (with augmented data)
        idata_new = run_inference(
            T_aug,
            p_i_aug,
            y_augmented,
            num_samples=4000,
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
    x_augmented = jnp.vstack([x_old, x_new.reshape(1, -1)])
    y_augmented = jnp.vstack([y_old, y_hyp])

    # Extract conditions
    T_aug = x_augmented[:, 0:1]
    p_aug = x_augmented[:, 1:2]
    c2h_aug = x_augmented[:, 2:3]
    c2m_aug = x_augmented[:, 3:4]
    c2w_aug = x_augmented[:, 4:5]
    c2i_aug = x_augmented[:, 5:6]

    # Calculate partial pressures
    p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, c2m_aug, c2w_aug, c2i_aug)

    # Run inference on augmented data
    idata_new = run_inference(
        T_aug,
        p_i_aug,
        y_augmented,
        num_samples=4000,
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
    args: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, xarray.DataTree, int, int, bool
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
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: xarray.DataTree,
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
    c2m_new = x_new[3:4].reshape(-1, 1)
    c2w_new = x_new[4:5].reshape(-1, 1)
    c2i_new = x_new[5:6].reshape(-1, 1)

    # Calculate partial pressures
    p_i_new = calculate_partial_pressures(p_new, c2h_new, c2m_new, c2w_new, c2i_new)

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
            jnp.array(y_new_samples[i].reshape(1, 1)),
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


def sp_find_opt_candidate(
    n_iterations: int,
    proposals: jnp.ndarray,
    r_obs: jnp.ndarray,
    idata: xarray.DataTree,
    n_hyp_samples: int,
    n_workers: int,
    filename: str,
    use_vi: bool = True,
):
    """
    Sequential optimization to find optimal experimental candidate.

    Args:
        n_iterations: Number of optimization iterations
        proposals: Current experimental conditions (n_current, 5)
        r_obs: Current observations (n_current, 1)
        idata: Current posterior samples
        n_hyp_samples: Number of hypothetical observations per candidate
        n_workers: Number of parallel workers
        filename: Path to save optimization results
        use_vi: If True, use VI; if False, use MCMC

    Returns:
        Tuple of (best_parameters, expected_KLD, index, name)
    """
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
    client.configure_optimization(objective="KLD")
    for _ in range(n_iterations):
        trials = client.get_next_trials(max_trials=1)
        for trial_idx, parameters in trials.items():
            candidate = jnp.array(
                [
                    parameters["T"],
                    parameters["p"],
                    parameters["c2h"],
                    parameters["c2m"],
                    parameters["c2w"],
                    parameters["c2i"],
                ],
            )
            KLD_vals = mp_KL_divergence(
                proposals,
                r_obs,
                candidate.squeeze(),
                idata,
                n_hyp_samples=n_hyp_samples,
                n_cores=n_workers,
                use_vi=use_vi,
            )
            mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
            std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"KLD": (mean_KLD, std_KLD)},
            )
            print(
                f"Trial {trial_idx} at proposal {candidate.tolist()} has KLD {mean_KLD} ± {std_KLD}"
            )
    client.summarize()
    client.save_to_json_file(filename)
    best_parameters, expected_KLD, index, name = client.get_best_parameterization()
    return best_parameters, expected_KLD, index, name


def mp_find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    proposals: jnp.ndarray,
    r_obs: jnp.ndarray,
    idata: xarray.DataTree,
    n_hyp_samples: int,
    n_workers: int,
    filename: Path,
    use_vi: bool = True,
):
    """
    Parallel optimization to find optimal experimental candidate.

    Evaluates multiple candidates in parallel within each iteration.

    Args:
        n_iterations: Number of optimization iterations
        n_trials_per_iter: Number of candidates to evaluate in parallel per iteration
        proposals: Current experimental conditions (n_current, 6)
        r_obs: Current observations (n_current, 1)
        idata: Current posterior samples
        n_hyp_samples: Number of hypothetical observations per candidate
        n_workers: Number of parallel workers for KL computation
        filename: Path to save optimization results
        use_vi: If True, use VI; if False, use MCMC

    Returns:
        Tuple of (best_parameters, expected_KLD, index, name)
    """
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
    client.configure_optimization(objective="KLD")
    for _ in range(n_iterations):
        trials = client.get_next_trials(max_trials=n_trials_per_iter)
        candidates = []
        indeces = []
        for trial_idx, parameters in trials.items():
            candidate = jnp.array(
                [
                    parameters["T"],
                    parameters["p"],
                    parameters["c2h"],
                    parameters["c2m"],
                    parameters["c2w"],
                    parameters["c2i"],
                ],
            )
            candidates.append(candidate)
            indeces.append(trial_idx)

        # Prepare arguments for parallel execution
        args_list = [
            (proposals, r_obs, candidate, idata, n_hyp_samples, n_workers, use_vi)
            for candidate in candidates
        ]

        with NestablePool(len(candidates)) as pool:
            results = pool.map(_candidate_kl_worker, args_list)

        for trial_idx, KLD_vals in zip(indeces, results):
            mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
            std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"KLD": (mean_KLD, std_KLD)},
            )
            print(f"Trial {trial_idx} has KLD {mean_KLD} ± {std_KLD}")
    client.summarize()
    client.save_to_json_file(str(filename))
    best_parameters, expected_KLD, index, name = client.get_best_parameterization()
    return best_parameters, expected_KLD, index, name


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
    print("BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  KL samples: {n_hyp_samples}")
    print(f"  Design iterations: {n_iterations}")
    print(f"  Optimization iterations: {n_opt_iterations}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    # Step 1: Sample prior and find initial optimal experiment
    print("\n[STEP 0] Sampling from prior and finding initial experiment...")
    prior_idata = sample_prior(num_samples=5000)

    # Use prior as "posterior" for initial optimization (no data yet)
    prior_idata.posterior = prior_idata.prior

    # Find first experiment from prior (no observations yet)
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
        filename=save_path / "ax_client_000.json",
        use_vi=True,
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

    print(
        f"Initial proposal: T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar"
    )
    print(f"Expected KLD: {expected_KLD['KLD'][0]:.4e}\n")

    # Step 2: Sequential design loop
    trials_sum = []

    for i in range(1, n_iterations + 1):
        print(f"\n{'='*70}")
        print(f"[ITERATION {i}/{n_iterations}]")
        print(f"{'='*70}")

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

        # Run Bayesian inference (Variational Inference for speed)
        print("Running variational inference...")
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs,
            num_samples=4000,
            num_warmup=4000,
            num_chains=8,
        )

        # Add prior samples for comparison
        # print("Adding prior samples to inference data...")
        # prior_samples = sample_prior(num_samples=4000)
        # idata.add_groups(prior=prior_samples.prior)

        # Save inference results
        trace_filename = save_path / f"trace_{i:03d}.nc"
        idata.to_netcdf(trace_filename)
        print(f"Saved trace to: {trace_filename.name}")

        # Find next optimal experiment via KL divergence maximization
        print(f"Optimizing next experiment (KL divergence maximization)...")
        best_parameters, expected_KLD, _, _ = mp_find_opt_candidate(
            n_iterations=n_opt_iterations,
            n_trials_per_iter=n_trials_per_iter,
            proposals=proposals,
            r_obs=r_obs,
            idata=idata,
            n_hyp_samples=n_hyp_samples,
            n_workers=num_workers,
            filename=save_path / f"ax_client_{i:03d}.json",
            use_vi=True,
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

        print(f"Next optimal experiment:")
        print(
            f"  T={best_parameters['T']:.1f}K, p={best_parameters['p']:.1f}bar,"
            f"c2h={best_parameters['c2h']:.2f}, c2m={best_parameters['c2m']:.1f}, "
            f"c2w={best_parameters['c2w']:.1f}, c2i={best_parameters['c2i']:.2f}"
        )
        print(f"  Expected KLD: {expected_KLD['KLD'][0]:.4e}")

        # Store iteration summary
        trial_summary = {
            "iteration": i,
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
    print("SEQUENTIAL DESIGN COMPLETE")
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
