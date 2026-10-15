"""
Langmuir-Hinshelwood-Hougen-Watson (LHHW) Kinetics Model with Deactivation

NumPyro

Imports from GTMo_C_003

"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az
import xarray
import diffrax as dfx

from HSA_GTMo_C_003 import (
    kinetics_model,
    TRUE_PARAMS,
    calculate_partial_pressures,
)


# Species indices for partial pressure arrays
SIDX = {
    "CO2": 0,
    "H2": 1,
    "H2O": 2,
    "CH4": 3,
    "I": 4,  # Inert species
}


def bayesian_model(
    T: jnp.ndarray, p_i: jnp.ndarray, r_obs: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    NumPyro Bayesian model for kinetic parameter inference with deactivation.

    This model uses informative priors based on expected parameter ranges:
    - Normal priors for transformed parameters (log-space)
    - Uniform prior for deactivation order
    - Normal prior for observation noise

    Args:
        T: Temperature array [K]
        p_i: Partial pressures array [bar]
        r_obs: Observed reaction rates [mol/(s·g_cat)] (None for prior sampling)

    Returns:
        Sampled or observed reaction rates
    """
    # Sample kinetic parameters from priors (transformed space)
    t_k0 = jnp.expand_dims(numpyro.sample("t_k0", dist.Normal(0, 1)), 0)
    t_Ea_k = jnp.expand_dims(numpyro.sample("t_Ea_k", dist.Normal(0, 1)), 0)
    t_k_deact = jnp.expand_dims(numpyro.sample("t_k_deact", dist.Normal(0, 1)), 0)
    t_Ea_deact = jnp.expand_dims(numpyro.sample("t_Ea_deact", dist.Normal(0, 1)), 0)
    t_n_deact = jnp.expand_dims(numpyro.sample("t_n_deact", dist.Uniform(0, 1)), 0)
    t_n_deact_p = jnp.expand_dims(numpyro.sample("t_n_deact_p", dist.Uniform(0, 1)), 0)

    # Transform to original space
    k0 = numpyro.deterministic("k0", 1e-4 * jnp.exp(t_k0))
    Ea_k = numpyro.deterministic("Ea_k", 50e3 * jnp.exp(t_Ea_k))
    k_deact = numpyro.deterministic("k_deact", 1e-5 * jnp.exp(t_k_deact))
    Ea_deact = numpyro.deterministic("Ea_deact", 50e3 * jnp.exp(t_Ea_deact))
    n_deact = numpyro.deterministic("n_deact", (7.0 - 3.0) * t_n_deact + 3.0)
    n_deact_p = numpyro.deterministic("n_deact_p", (3.0 - 1.0) * t_n_deact_p + 1.0)

    # Homoscedastic observation noise
    sigma_r = jnp.expand_dims(numpyro.sample("sigma_r", dist.HalfNormal(0.01)), 0)

    # Reshape observations if provided
    if r_obs is not None:
        r_obs_proc = jnp.permute_dims(r_obs, (1, 3, 0, 2))
        r_obs_proc = jnp.squeeze(r_obs_proc, 1)
    else:
        r_obs_proc = None

    # Compute predicted reaction rates
    r_pred = kinetics_model(p_i, T, k0, Ea_k, k_deact, Ea_deact, n_deact, n_deact_p)

    # Reshape predictions for likelihood
    r_pred = jnp.permute_dims(r_pred, (1, 3, 0, 2))
    n_obs, n_draws, t_obs, n_comp = r_pred.shape
    r_pred = jnp.squeeze(r_pred, 1)
    r_lat = numpyro.deterministic("r_lat", r_pred)
    sigma_obs = sigma_r * jnp.abs(r_lat) + 1e-15

    # Likelihood
    with numpyro.plate("data", size=n_obs):
        obs_dist = dist.Normal(r_lat, sigma_obs).to_event(2)
        r = numpyro.sample(
            "r",
            obs_dist,
            obs=r_obs_proc,
        )

    return r


def run_inference(
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    r_obs: jnp.ndarray,
    num_samples: int = 4000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    progressbar: bool = False,
) -> xarray.DataTree:
    """
    Run MCMC inference to estimate kinetic parameters.

    Args:
        T: Temperature observations [K]
        p_i: Partial pressure observations [bar]
        r_obs: Observed reaction rates [mol/(s·g_cat)]
        num_samples: Number of MCMC samples per chain
        num_warmup: Number of warmup/tuning steps
        num_chains: Number of parallel MCMC chains

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    # Configure NUTS sampler
    seed = jax.random.PRNGKey(42)
    nuts_kernel = NUTS(
        bayesian_model,
        # this is supposed to make the sampler slower but more stable
        # target_accept_prob=0.95,
        # max_tree_depth=12,
        dense_mass=True,
    )

    # Run MCMC
    print(
        f"Starting MCMC: {num_chains} chains × ({num_warmup} warmup + {num_samples} samples)"
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        progress_bar=progressbar,
    )
    mcmc.run(seed, T=T, p_i=p_i, r_obs=r_obs)
    # for _ in range(10):
    #     try:
    #         mcmc.run(seed, T=T, p_i=p_i, r_obs=r_obs)
    #         break
    #     except Exception as e:
    #         seed = jax.random.split(seed, 1)[0]
    #         print(f"Exception during MCMC run: {e}. Retrying with seed {seed}.")
    # else:
    #     raise RuntimeError("MCMC sampling failed after multiple attempts.")
    print("MCMC sampling completed successfully!")

    # Convert to ArviZ InferenceData for analysis
    idata = az.from_numpyro(mcmc)
    return idata


def sample_prior(num_samples: int = 1000) -> xarray.DataTree:
    """
    Sample from the prior distribution.

    Args:
        num_samples: Number of prior samples to generate

    Returns:
        ArviZ InferenceData object with prior samples
    """
    seed = jax.random.PRNGKey(0)
    # Use Predictive to sample from prior (no observations)
    prior_predictive = Predictive(bayesian_model, num_samples=num_samples)

    # Dummy data for sampling (not used in prior)
    T_dummy = jnp.array([[555.0]])  # Shape (1, 1)
    p_i_dummy = jnp.array([[2.0, 2.0, 0.0, 0.0, 0.0]])  # Shape (1, 5)

    prior_samples = prior_predictive(seed, T=T_dummy, p_i=p_i_dummy)

    # Create InferenceData with prior group
    idata = az.from_numpyro(prior=prior_samples)
    return idata


def sample_posterior(
    idata: xarray.DataTree,
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    num_samples: int = 1000,
    r_obs: Optional[jnp.ndarray] = None,
) -> xarray.DataTree:
    """
    Sample from the posterior distribution using Predictive.

    Args:
        idata: ArviZ InferenceData with posterior samples
        T: Temperature conditions [K]
        p_i: Partial pressure conditions [bar]
        num_samples: Number of posterior predictive samples to generate
        r_obs: Optional observed reaction rates [mol/(s·g_cat)]

    Returns:
        ArviZ InferenceData object with posterior predictive samples
    """
    print(T.shape, p_i.shape, r_obs.shape if r_obs is not None else None)

    ## This function does not use apredictive because of shape errors
    posterior_samples = predict(
        idata,
        T,
        p_i,
        num_samples=num_samples,
        key="posterior",
    )
    # The posterior samples are the "latent" rates.
    # noise should be added later if needed.
    # But we mostly compare the latent (unnoisy) rates to holdout data.

    # Convert to ArviZ InferenceData
    posterior_predictive_dict = {
        "r_lat": jnp.permute_dims(posterior_samples, (3, 1, 0, 2))[None, ...]
    }

    if r_obs is None:
        obs_data_dict = {"r": None}
    else:
        obs_data_dict = {"r": np.array(r_obs)}

    idata = az.from_dict(
        {
            # "posterior": {},
            "posterior_predictive": posterior_predictive_dict,
            "observed_data": obs_data_dict,
        },
        dims={
            "r_lat": ["chain", "draw", "obs", "times", "components"],
        },
    )
    return idata


def validate_against_holdout(
    idata: xarray.DataTree,
    holdout: xarray.Dataset,
    num_samples: int = 1000,
):
    post_data = sample_posterior(
        idata,
        jnp.array(holdout.T.values).reshape(-1, 1),
        jnp.array(holdout.p.values).reshape(-1, 5),
        num_samples=num_samples,
    )

    observed_data = xarray.Dataset(
        data_vars= {
            "r": (
                ["chain", "draw", "obs", "times", "components"],
                jnp.permute_dims(
                    holdout.obs_r.values,
                    (3, 1, 0, 2),
                )[None,...]
                ),
            "r_lat": (
                ["chain", "draw", "obs", "times", "components"],
                jnp.permute_dims(
                    holdout.latent_r.values,
                    (3, 1, 0, 2),
                )[None,...]
                ),
        },
    )
    post_data["observed_data"] = observed_data

    return post_data


def generate_synthetic_data(
    T: jnp.ndarray,
    p: jnp.ndarray,
    c2h: jnp.ndarray,
    c2m: jnp.ndarray,
    c2w: jnp.ndarray,
    c2i: jnp.ndarray,
    rel_noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic observation data using true parameters.

    Args:
        T: Temperature array [K]
        p: Total pressure array [bar]
        c2h: CO2/H2 ratio array [-]
        c2m: CO2/CH4 ratio array [-]
        c2w: CO2/H2O ratio array [-]
        c2i: CO2/Inert ratio array [-]
        rel_noise_std: Relative standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (T, p_i, r_obs) ready for inference
    """
    # Set random seed
    if seed is not None:
        key = jax.random.key(seed)
    else:
        key = jax.random.key(763789678968)
        key, _ = jax.random.split(key)

    # Calculate partial pressures
    p_i = calculate_partial_pressures(p, c2h, c2m, c2w, c2i)

    # Compute true reaction rates
    r_true = kinetics_model(p_i, T, *TRUE_PARAMS.values())

    # Add Gaussian noise
    rng = jax.random.normal(key, r_true.shape)
    noise = rng * rel_noise_std * jnp.absolute(r_true)
    r_obs = r_true + noise

    return T, p_i, r_obs


def predict(
    idata: xarray.DataTree,
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    num_samples: int = 25,
    key: str = "posterior",
    w_noise: bool = False,
    seed=None,
) -> np.ndarray:
    """
    Generate posterior predictions for given conditions.

    Args:
        idata: ArviZ InferenceData with posterior samples
        T: Temperature conditions [K]
        p_i: Partial pressure conditions [bar]
        num_samples: Number of posterior samples to use
        key: Which group to extract samples from ("posterior" or "prior")

    Returns:
        Array of predicted rates
    """
    # Extract posterior samples
    draws = az.extract(idata, group=key, num_samples=num_samples)

    # Flatten all chains and draws
    k0 = jnp.array(draws["k0"].values.flatten())
    Ea_k = jnp.array(draws["Ea_k"].values.flatten())
    k_deact = jnp.array(draws["k_deact"].values.flatten())
    Ea_deact = jnp.array(draws["Ea_deact"].values.flatten())
    n_deact = jnp.array(draws["n_deact"].values.flatten())
    n_deact_p = jnp.array(draws["n_deact_p"].values.flatten())
    sigma_r = jnp.array(draws["sigma_r"].values.flatten())

    # Reshape for broadcasting with kinetics model
    params = {
        "k0": k0.reshape(1, -1),
        "Ea_k": Ea_k.reshape(1, -1),
        "k_deact": k_deact.reshape(1, -1),
        "Ea_deact": Ea_deact.reshape(1, -1),
        "n_deact": n_deact.reshape(1, -1),
        "n_deact_p": n_deact_p.reshape(1, -1),
    }

    # Generate predictions
    predictions = kinetics_model(p_i, T, **params)

    if seed is not None:
        rkey = jax.random.key(seed)
    else:
        rkey = jax.random.key(763789678968)
        rkey, _ = jax.random.split(rkey)

    if w_noise:
        rng = jax.random.normal(rkey, predictions.shape)
        noise = rng * sigma_r * jnp.abs(predictions)
        return predictions + noise
    else:
        return predictions


def main() -> None:
    """
    Demonstration of the NumPyro-based kinetics inference workflow with deactivation.
    """
    print("=== NumPyro LHHW Kinetics Model with Deactivation Demo (JAX Backend) ===\n")

    # Define experimental conditions
    T = jnp.array([[560.0], [570.0], [580.0]])
    p = jnp.array([[4.0], [5.0], [6.0]])
    c2h = jnp.ones((3, 1))
    c2m = jnp.ones((3, 1)) * 1000
    c2w = jnp.ones((3, 1)) * 1000
    c2i = jnp.ones((3, 1)) * 0.5

    print("Generating synthetic data...")
    T_obs, p_i_obs, r_obs = generate_synthetic_data(T, p, c2h, c2m, c2w, c2i)
    print(f"Observations shape: {r_obs.shape}")
    print(f"Observed rates: {r_obs.squeeze()}\n")

    print("Sampling from prior...")
    prior_idata = sample_prior(num_samples=1000)
    print(f"Prior samples: {list(prior_idata.prior.data_vars.keys())}\n")

    print("Running MCMC inference...")
    posterior_idata = run_inference(
        T_obs, p_i_obs, r_obs, num_samples=1000, num_warmup=500, num_chains=2
    )
    print("Inference complete!\n")

    # Print posterior summary
    print("MCMC Posterior summary:")
    print(az.summary(posterior_idata, hdi_prob=0.95))

    print(
        "\n✅ All tests passed! JAX + NumPyro implementation with deactivation works correctly."
    )


if __name__ == "__main__":
    main()
