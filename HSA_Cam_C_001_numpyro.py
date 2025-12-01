"""
Benchmark Experimental Design using Latin hypercube sampling (numpyro Version)

"""

import argparse

import json
from pathlib import Path
import arviz as az
import numpy as np
from scipy.stats import gaussian_kde, qmc
from scipy.integrate import simpson
import xarray
import namer
from NestPools import NestablePool
import numpyro

numpyro.set_host_device_count(8)
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit",True)

from HSA_Mo_C_001_numpyro import (
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
    default=2,
    # default=12,
    help="Number of workers for parallel processing KLD computations",
)
parser.add_argument(
    "--KLsamples",
    type=int,
    # default=12,
    default=2,
    help="Number of hypothetical samples for KL divergence estimation",
)
parser.add_argument(
    "--iterations",
    type=int,
    # default=8,
    default=32,
    help="Number of points to additionally sample in the experiment",
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


def _single_kl_worker(args):
    """
    Worker function for computing KL divergence for a single hypothetical observation.
    Must be at top level for multiprocessing on Windows.
    """
    x_old, y_old, x_new, idata_old, y_hyp = args

    # Create augmented dataset
    x_augmented = jnp.vstack([x_old, x_new.reshape(1, -1)])
    y_augmented = jnp.concatenate([y_old, y_hyp], axis=1)  # (10, n_old+1, 1, 1)

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
        num_samples=400,  # TODO 4000
        num_warmup=100,  # TODO 1000
        num_chains=2,  # TODO
    )

    # Compute KL divergence
    klds_i = kl_divergence_between_models(
        idata_new,
        idata_old,
        key_new="posterior",
        key_old="posterior",
        n_samples=100,  # TODO 1000
        var_names=[
            "k0",
            "Ea_k",
            "k_deact",
            "Ea_deact",
            "n_deact",
            "sigma_r",
        ],
    )
    return klds_i


def _candidate_kl_worker(
    args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, az.InferenceData, int, int],
):
    """Top-level worker for parallelizing over candidate proposals."""
    x_old, y_old, x_new, idata_old, n_hyp_samples, n_cores = args
    return mp_KL_divergence(
        x_old=x_old,
        y_old=y_old,
        x_new=x_new.squeeze(),
        idata_old=idata_old,
        n_hyp_samples=n_hyp_samples,
        n_cores=n_cores,
    )


def mp_KL_divergence(
    x_old: jnp.ndarray,
    y_old: jnp.ndarray,
    x_new: jnp.ndarray,
    idata_old: az.InferenceData,
    n_hyp_samples: int = 5,
    n_cores: int = 4,
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
    )  # should be (10, 1, 1, n_hyp_samples)

    print(f"Generated {n_hyp_samples} hypothetical observations at new condition")
    print(y_new_samples.shape)

    # Step 2: Compute KL divergence for each hypothetical observation in parallel
    args_list = [
        (
            x_old,
            y_old,
            x_new,
            idata_old,
            jnp.expand_dims(y_new_samples[:, :, :, i], axis=-1),  # (10, 1, 1, *1*)
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
                "k_deact",
                "Ea_deact",
                "n_deact",
                "sigma_r",
            ],
        },
        data=np.stack(results, axis=0),
    )

    # Step 4: Return KL divergences across all hypothetical samples
    return klds


def LHS_data(n_samples: int):
    """
    Generate Latin Hypercube Samples for experimental design.

    Args:
        n_samples: Number of samples to generate
    Returns:
    jnp.ndarray: Array of shape (n_samples, 6) with sampled experimental conditions
    """
    T_range = jnp.array([450, 600])
    p_range = jnp.array([3, 9])
    c2h_range = jnp.array([0.2, 4])
    c2m_range = jnp.array([0.5, 1000])
    c2w_range = jnp.array([0.5, 1000])
    c2i_range = jnp.array([0.05, 1])

    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=n_samples)

    # Scale samples to the appropriate ranges
    T_lhs = qmc.scale(sample[:, 0:1], T_range[0], T_range[1])
    p_lhs = qmc.scale(sample[:, 1:2], p_range[0], p_range[1])
    c2h_lhs = qmc.scale(sample[:, 2:3], c2h_range[0], c2h_range[1])
    c2m_lhs = qmc.scale(sample[:, 3:4], c2m_range[0], c2m_range[1])
    c2w_lhs = qmc.scale(sample[:, 4:5], c2w_range[0], c2w_range[1])
    c2i_lhs = qmc.scale(sample[:, 5:6], c2i_range[0], c2i_range[1])

    proposals = jnp.column_stack([T_lhs, p_lhs, c2h_lhs, c2m_lhs, c2w_lhs, c2i_lhs])
    return proposals


def main(args):
    """
    Run sequential Bexperimental design loop.

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

    print("=" * 70)
    print("BAYESIAN OPTIMAL EXPERIMENTAL DESIGN - SEQUENTIAL LOOP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  KL samples: {n_hyp_samples}")
    print(f"  Design iterations: {n_iterations}")
    print(f"  Experiment ID: {EXP_ID}")
    print("=" * 70)

    all_proposals = LHS_data(n_samples=n_iterations)

    # Step 1: Sample prior and find initial optimal experiment
    print("\n[STEP 0] Sampling from prior ")
    # prior_idata = sample_prior(num_samples=4000)
    # prior_idata.add_groups(posterior=prior_idata.prior)

    proposals = jnp.empty((0, 6))
    r_obs = jnp.empty((10, 0, 1, 1))

    # Step 2: Sequential design loop
    trials_sum = []

    for i in range(0, n_iterations):
        print(f"\n{'='*70}")
        print(f"[ITERATION {i}/{n_iterations}]")
        print(f"{'='*70}")

        new_proposal = all_proposals[i : i + 1, :]

        # KLD_vals = mp_KL_divergence(
        #     x_old=proposals,
        #     y_old=r_obs,
        #     x_new=new_proposal.squeeze(),
        #     idata_old=prior_idata,
        #     n_hyp_samples=n_hyp_samples,
        #     n_cores=num_workers,
        # )
        # mean_KLD = KLD_vals.sum(axis=1).mean(axis=0).item()
        # std_KLD = KLD_vals.sum(axis=1).std(axis=0).item()

        print(f"New proposal: {new_proposal}")
        # print(f"Expected KLD: {mean_KLD:.4f} ± {std_KLD:.4f}")

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

        print("Running MCMC inference...")
        idata = run_inference(
            T=T,
            p_i=p_i,
            r_obs=r_obs,
            num_samples=1000,  # TODO
            num_warmup=4000,
            num_chains=4,
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
                "expected_KLD": None,  # mean_KLD,
                "std_KLD": None,  # std_KLD,
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
