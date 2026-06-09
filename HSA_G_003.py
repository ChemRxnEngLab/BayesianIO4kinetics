"""
Bayesian Optimal Experimental Design using KL Divergence (Numpyro Version)

This module implements Bayesian optimal experimental design for kinetic parameter
estimation using KL divergence as the utility function. Refactored to work with
Numpyro probabilistic programming framework (jax backend).

Key Features:
- Homoscedastic noise modeling
- TWO acquisition functions:
  1. Parametric KLD: KL divergence between parameter posteriors
  2. Predictive KLD (BALD): KL divergence between predictive distributions
- Parallel experimental design optimization
- Support for MCMC (accurate)
- Integration with Ax platform for Bayesian optimization

Main Functions:
- kl_marginal: Compute KL divergence between marginal distributions using KDE
- kl_divergence_between_models: Compute KL divergence for all parameters
- mp_parametric_KL_divergence: Parallel parametric KL computation using multiprocessing
- mp_predictive_KL_divergence: Parallel predictive KL computation using multiprocessing
- mp_find_opt_candidate: Parallel Bayesian optimization of experimental design

Design Philosophy:
- Use VI (variational inference) for rapid exploration
- Use MCMC for final accurate inference
- Parallelize across hypothetical observations for speedup
- Use Ax platform for adaptive experimental design
"""

import json
import argparse
from pathlib import Path
from typing import Optional
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


from HSA_Mo_B_012_numpyro import (
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


def _single_predictive_kl_worker(args):
    """
    Worker function for computing predictive KL divergence for a single hypothetical observation.
    Must be at top level for multiprocessing on Windows.
    """
    x_old, y_old, x_new, x_test, idata_old, y_hyp, n_pred_samples, use_vi = args

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
    p_i_aug = calculate_partial_pressures(p_aug, c2h_aug, c2m_aug, c2w_aug, c2i_aug)

    # Run inference on augmented data
    idata_new = run_inference(
        T_aug,
        p_i_aug,
        y_augmented.squeeze(),
        num_samples=n_pred_samples,
        num_warmup=1000,
        num_chains=2,
    )

    # Extract test conditions
    T_test = x_test[:, 0:1]
    p_test = x_test[:, 1:2]
    c2h_test = x_test[:, 2:3]
    c2m_test = x_test[:, 3:4]
    c2w_test = x_test[:, 4:5]
    c2i_test = x_test[:, 5:6]
    p_i_test = calculate_partial_pressures(
        p_test, c2h_test, c2m_test, c2w_test, c2i_test
    )

    # Generate predictions with new posterior
    pred_new = predict(
        idata_new,
        T_test,
        p_i_test,
        key="posterior",
        num_samples=n_pred_samples,
    )

    # Generate predictions with old posterior (passed as part of args would be better,
    # but for now we'll recompute - could optimize this)
    pred_old = predict(
        idata_old,
        T_test,
        p_i_test,
        key="posterior",
        num_samples=n_pred_samples,
    )

    # Compute KL divergence at each test point
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
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    n_pred_samples: int = 500,
    n_cores: int = 4,
    use_vi: bool = True,
):
    """
    Compute predictive KL divergence in parallel across multiple hypothetical observations.

    This is the parallelized version of predictive_KL_divergence() for faster computation.

    Args:
        x_old: Current experimental conditions (n_current, 5)
        y_old: Current observations (n_current, 1)
        x_new: New experimental condition to test (5,)
        x_test: Test conditions to measure prediction changes (n_test, 5)
        idata_old: Current posterior samples
        n_hyp_samples: Number of hypothetical observations
        n_pred_samples: Number of samples for predictive distributions
        n_cores: Number of parallel workers
        use_vi: If True, use VI (fast); if False, use MCMC (accurate)

    Returns:
        xarray.DataArray of KL divergences with dimensions (hyp_sample, test_point)
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

    # Step 1: Generate hypothetical observations at x_new
    y_new_samples = predict(
        idata_old,
        T_new,
        p_i_new,
        key="posterior",
        num_samples=n_hyp_samples,
    ).reshape(-1, 1)
    print(f"Generated {n_hyp_samples} hypothetical observations at candidate location")

    # Step 2: Compute predictive KL in parallel
    args_list = [
        (
            x_old,
            y_old,
            x_new,
            x_test,
            idata_old,
            jnp.array(y_new_samples[i].reshape(1, 1)),
            n_pred_samples,
            use_vi,
        )
        for i in range(n_hyp_samples)
    ]

    n_procs = min(n_cores, n_hyp_samples)
    with NestablePool(n_procs) as pool:
        results = pool.map(_single_predictive_kl_worker, args_list)
    print("Completed predictive KL divergence computations in parallel")

    # Package results
    n_test = x_test.shape[0]
    klds = xarray.DataArray(
        dims=["hyp_sample", "test_point"],
        coords={
            "hyp_sample": np.arange(n_hyp_samples),
            "test_point": np.arange(n_test),
        },
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
    """
    Generate a Latin Hypercube test grid for measuring predictive changes.

    Args:
        n_points: Number of test points
        T_range: Temperature range [K]
        p_range: Pressure range [bar]
        c2h_range: C2H4 concentration range
        f_range: Flow rate range
        c2i_range: C2H6 inlet concentration range

    Returns:
        torch.Tensor of shape (n_points, 5) with test conditions
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=n_points)

    # Scale to bounds
    lower_bounds = np.array(
        [
            T_range[0],
            p_range[0],
            c2h_range[0],
            c2m_range[0],
            c2w_range[0],
            c2i_range[0],
        ],
    )
    upper_bounds = np.array(
        [
            T_range[1],
            p_range[1],
            c2h_range[1],
            c2m_range[1],
            c2w_range[1],
            c2i_range[1],
        ]
    )

    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

    return jnp.array(scaled_sample)


def _single_parametric_kl_worker(args):
    """
    Worker function for computing parametric KL divergence for a single hypothetical observation.
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
    # Accurate MCMC inference
    idata_new = run_inference(
        T_aug,
        p_i_aug,
        y_augmented.squeeze(),
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


def mp_parametric_KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    n_cores: int = 4,
    use_vi: bool = True,
):
    """
    Compute parametric KL divergence in parallel across multiple hypothetical observations.

    This measures changes in parameter posteriors.

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
        results = pool.map(_single_parametric_kl_worker, args_list)
    print("Completed parametric KL divergence computations in parallel")

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


def mp_find_opt_candidate(
    n_iterations: int,
    n_trials_per_iter: int,
    proposals: jnp.ndarray,
    r_obs: jnp.ndarray,
    idata: az.InferenceData,
    n_hyp_samples: int,
    n_workers: int,
    filename: str,
    use_vi: bool = True,
    acquisition: str = "parameter",
    x_test: Optional[jnp.ndarray] = None,
    n_pred_samples: int = 500,
):
    """
    Parallel optimization to find optimal experimental candidate.

    Evaluates multiple candidates in parallel within each iteration.

    Args:
        n_iterations: Number of optimization iterations
        n_trials_per_iter: Number of candidates to evaluate in parallel per iteration
        proposals: Current experimental conditions (n_current, 5)
        r_obs: Current observations (n_current, 1)
        idata: Current posterior samples
        n_hyp_samples: Number of hypothetical observations per candidate
        n_workers: Number of parallel workers for KL computation
        filename: Path to save optimization results
        use_vi: If True, use VI; if False, use MCMC
        acquisition: "parametric" for parametric KLD, "predictive" for predictive KLD
        x_test: Test grid for predictive KLD (required if acquisition="predictive")
        n_pred_samples: Number of samples for predictive distributions

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

        # Evaluate candidates based on acquisition function
        results = []
        for candidate in candidates:
            if acquisition == "parametric":
                KLD_vals = mp_parametric_KL_divergence(
                    x_old=proposals,
                    y_old=r_obs,
                    x_new=candidate,
                    idata_old=idata,
                    n_hyp_samples=n_hyp_samples,
                    n_cores=n_workers,
                    use_vi=use_vi,
                )
            elif acquisition == "predictive":
                if x_test is None:
                    raise ValueError(
                        "x_test must be provided for predictive acquisition"
                    )
                KLD_vals = mp_predictive_KL_divergence(
                    x_old=proposals,
                    y_old=r_obs,
                    x_new=candidate,
                    x_test=x_test,
                    idata_old=idata,
                    n_hyp_samples=n_hyp_samples,
                    n_pred_samples=n_pred_samples,
                    n_cores=n_workers,
                    use_vi=use_vi,
                )
                results.append(KLD_vals)
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition}")

        for trial_idx, KLD_vals in zip(indeces, results):
            mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
            std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()
            client.complete_trial(
                trial_index=trial_idx,
                raw_data={"KLD": (mean_KLD, std_KLD)},
            )
            print(f"Trial {trial_idx} has KLD {mean_KLD} ± {std_KLD}")
    client.summarize()
    client.save_to_json_file(filename)
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
    acquisition = args.acquisition
    n_test_points = args.ntestpoints

    print("=" * 70)
    print("BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  KL samples: {n_hyp_samples}")
    print(f"  Design iterations: {n_iterations}")
    print(f"  Optimization iterations: {n_opt_iterations}")
    print(f"  Acquisition function: {acquisition}")
    if acquisition == "predictive":
        print(f"  Test points: {n_test_points}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    # Generate test grid for predictive KLD
    x_test = None
    if acquisition == "predictive":
        print("\n[SETUP] Generating test grid for predictive KLD...")
        x_test = generate_test_grid(n_points=n_test_points)
        print(f"Generated {n_test_points} test points")

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
        filename=str(save_path / "ax_client_000.json"),
        use_vi=True,
        acquisition=acquisition,
        x_test=x_test,
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
            T,
            p,
            c2h,
            c2m,
            c2w,
            c2i,
            seed=42,
            rel_noise_std=0.1,
        )
        print(f"Observed rates: min={r_obs.min():.2e}, max={r_obs.max():.2e}")

        # Run Bayesian inference (Variational Inference for speed)
        print("Running variational inference...")
        # print("Running MCMC inference...")
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs.squeeze(),
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
            filename=str(save_path / f"ax_client_{i:03d}.json"),
            use_vi=True,
            acquisition=acquisition,
            x_test=x_test,
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
