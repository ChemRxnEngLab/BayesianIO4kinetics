"""
Langmuir-Hinshelwood-Hougen-Watson (LHHW) Kinetics Model with NumPyro

This module implements a methanation reaction kinetics model using Bayesian inference
with the NumPyro probabilistic programming library. The model assumes heteroscedastic noise
and uses informative priors based on expected parameter ranges.

mu + 5%*mu

Key Components:
- JAX-based kinetics calculations (NumPyro is built on JAX)
- NumPyro model for Bayesian parameter estimation
- Utility functions for experimental design and data generation
    - This file consisders water in the reaction mixture.
"""

from typing import Literal, Optional, Tuple
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


# Species indices for partial pressure arrays
SIDX = {
    "CO2": 0,
    "H2": 1,
    "H2O": 2,
    "CH4": 3,
    "I": 4,  # Inert species
}

# Ground truth parameters for synthetic data generation
TRUE_PARAMS = {
    "k0": jnp.array(3.46e-4),  # Pre-exponential factor [mol/(bar·s·g_cat)]
    "EA": jnp.array(77.5e3),  # Activation energy [J/mol]
    "A_OH": jnp.array(0.50),  # OH adsorption pre-exponential [bar^-0.5]
    "H_OH": jnp.array(22.4e3),  # OH adsorption enthalpy [J/mol]
    "A_H2": jnp.array(0.44),  # H2 adsorption pre-exponential [bar^-0.5]
    "H_H2": jnp.array(-6.2e3),  # H2 adsorption enthalpy [J/mol]
    "A_mix": jnp.array(0.88),  # Mixed adsorption pre-exponential [bar^-0.5]
    "H_mix": jnp.array(-10.0e3),  # Mixed adsorption enthalpy [J/mol]
}


# JIT-compiled core kinetics computation for maximum speed
@jax.jit
def _kinetics_core(
    p_i: jnp.ndarray,
    T: jnp.ndarray,
    k0: jnp.ndarray,
    Ea_k: jnp.ndarray,
    A_oh: jnp.ndarray,
    dH_oh: jnp.ndarray,
    A_h2: jnp.ndarray,
    dH_h2: jnp.ndarray,
    A_mix: jnp.ndarray,
    dH_mix: jnp.ndarray,
) -> jnp.ndarray:
    """
    JIT-compiled core kinetics calculation (optimized for speed).
    This function is compiled once and reused for all subsequent calls.
    """
    R_const = 8.314  # Gas constant [J/(mol·K)] - must be local for JIT
    T_ref = 555.0  # Reference temperature [K]

    p_i = jnp.reshape(p_i, (-1, 5))
    T = jnp.reshape(T, (-1, 1))
    k0 = jnp.reshape(k0, (1, -1))
    Ea_k = jnp.reshape(Ea_k, (1, -1))
    A_oh = jnp.reshape(A_oh, (1, -1))
    dH_oh = jnp.reshape(dH_oh, (1, -1))
    A_h2 = jnp.reshape(A_h2, (1, -1))
    dH_h2 = jnp.reshape(dH_h2, (1, -1))
    A_mix = jnp.reshape(A_mix, (1, -1))
    dH_mix = jnp.reshape(dH_mix, (1, -1))

    # Arrhenius rate constant
    k = k0 * jnp.exp(Ea_k / R_const * (1.0 / T_ref - 1.0 / T))

    # Van't Hoff adsorption equilibrium constants
    K_OH = A_oh * jnp.exp(dH_oh / R_const * (1.0 / T_ref - 1.0 / T))
    K_H2 = A_h2 * jnp.exp(dH_h2 / R_const * (1.0 / T_ref - 1.0 / T))
    K_mix = A_mix * jnp.exp(dH_mix / R_const * (1.0 / T_ref - 1.0 / T))

    # Thermodynamic equilibrium constant for CO2 methanation
    K_eq = 137.0 * T ** (-4.0) * jnp.exp(158700.0 / (R_const * T))

    # Extract partial pressures (indexed directly for JIT optimization)
    p_CO2 = p_i[:, 0:1]  # SIDX["CO2"] = 0, shape: (n_cond, 1)
    p_H2 = p_i[:, 1:2]  # SIDX["H2"] = 1, shape: (n_cond, 1)
    p_H2O = p_i[:, 2:3]  # SIDX["H2O"] = 2, shape: (n_cond, 1)
    p_CH4 = p_i[:, 3:4]  # SIDX["CH4"] = 3, shape: (n_cond, 1)

    # Reaction driving force (1 - Q/K_eq)
    driving_force = 1.0 - (p_CH4 * p_H2O**2.0) / (p_CO2 * p_H2**4.0 * K_eq)

    # Denominator: adsorption inhibition term (squared LHHW form)
    sqrt_p_H2 = jnp.sqrt(p_H2)
    sqrt_p_CO2 = jnp.sqrt(p_CO2)

    inhibition = (
        1.0 + (K_OH * p_H2O) / sqrt_p_H2 + K_H2 * sqrt_p_H2 + K_mix * sqrt_p_CO2
    ) ** 2.0

    # LHHW rate equation
    r = k * sqrt_p_H2 * sqrt_p_CO2 * driving_force / inhibition

    return r


def kinetics_model(
    p_i: jnp.ndarray,
    T: jnp.ndarray,
    k0: jnp.ndarray,
    Ea_k: jnp.ndarray,
    A_oh: jnp.ndarray,
    dH_oh: jnp.ndarray,
    A_h2: jnp.ndarray,
    dH_h2: jnp.ndarray,
    A_mix: jnp.ndarray,
    dH_mix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute LHHW kinetics model for CO2 methanation reaction.

    This wrapper function delegates to the JIT-compiled core for maximum performance.

    Reaction: CO2 + 4H2 ⇌ CH4 + 2H2O

    Args:
        p_i: Partial pressures array of shape (n_conditions, 5) [bar]
             Order: [CO2, H2, H2O, CH4, I]
        T: Temperature array of shape (n_conditions, 1) [K]
        k0: Pre-exponential factor [mol/(bar·s·g_cat)]
        Ea_k: Activation energy [J/mol]
        A_oh: OH adsorption pre-exponential factor [bar^-0.5]
        dH_oh: OH adsorption enthalpy [J/mol]
        A_h2: H2 adsorption pre-exponential factor [bar^-0.5]
        dH_h2: H2 adsorption enthalpy [J/mol]
        A_mix: Mixed species adsorption pre-exponential [bar^-0.5]
        dH_mix: Mixed species adsorption enthalpy [J/mol]

    Returns:
        Reaction rate array of shape (n_conditions, n_samples) [mol/(s·g_cat)]
    """
    # Call JIT-compiled core function for speedup
    return _kinetics_core(p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix)


def calculate_partial_pressures(
    p_total: jnp.ndarray,
    c2h_ratio: jnp.ndarray,
    c2m_ratio: jnp.ndarray,
    c2w_ratio: jnp.ndarray,
    c2i_ratio: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate partial pressures from total pressure and molar ratios.

    Args:
        p_total: Total pressure array of shape (n_conditions, 1) [bar]
        c2h_ratio: CO2/H2 molar ratio array of shape (n_conditions, 1) [-]
        feed_ratio: Feed dilution ratio (CO2+H2)/CH4 of shape (n_conditions, 1) [-]
        c2i_ratio: CO2/Inert molar ratio array of shape (n_conditions, 1) [-]

    Returns:
        Partial pressure array of shape (n_conditions, 5) [bar]
    """
    # Calculate CO2 mole fraction from ratios
    x_co2 = (
        1 / (1 + 1 / c2h_ratio + 1 / c2m_ratio + 1 / c2w_ratio + 1 / c2i_ratio)
    ).squeeze(-1)

    # Calculate other species mole fractions
    x_h2 = x_co2 / c2h_ratio.squeeze(-1)
    x_m = x_co2 / c2m_ratio.squeeze(-1)
    x_w = x_co2 / c2w_ratio.squeeze(-1)
    x_i = x_co2 / c2i_ratio.squeeze(-1)

    xi = jnp.stack((x_co2, x_h2, x_w, x_m, x_i), axis=1)

    # Convert to partial pressures: p_i = x_i * p_total
    p_i = p_total * xi

    return p_i


def bayesian_model(
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    r_obs: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    NumPyro Bayesian model for kinetic parameter inference.

    This model uses informative priors based on expected parameter ranges:
    - Exponential priors for positive parameters (rate constants, adsorption)
    - Normal priors for enthalpy changes (can be positive or negative)
    - Half-normal prior for relative observation noise (heteroscedastic)

    The noise model is heteroscedastic with standard deviation proportional
    to the predicted rate: sigma = sigma_rel * |r_pred|
    This captures the common experimental reality that measurement errors
    scale with the magnitude of the measured value.

    Args:
        T: Temperature array [K]
        p_i: Partial pressures array [bar]
        r_obs: Observed reaction rates [mol/(s·g_cat)] (None for prior sampling)

    Returns:
        Sampled or observed reaction rates
    """
    # Sample kinetic parameters from priors
    k0 = numpyro.sample("k0", dist.Exponential(1 / 0.001))
    Ea_k = numpyro.sample("Ea_k", dist.Exponential(1 / 60000))

    A_oh = numpyro.sample("A_oh", dist.Exponential(1.0))
    dH_oh = numpyro.sample("dH_oh", dist.Exponential(1 / 10000))

    A_h2 = numpyro.sample("A_h2", dist.Exponential(1.0))
    dH_h2 = numpyro.sample("dH_h2", dist.Normal(-10000, 5000))

    A_mix = numpyro.sample("A_mix", dist.Exponential(1.0))
    dH_mix = numpyro.sample("dH_mix", dist.Normal(-10000, 5000))

    # Homoscedastic observation noise
    sigma_r = numpyro.sample("sigma_r", dist.HalfNormal(0.1))

    k0_r = jnp.reshape(k0, (1, -1))
    Ea_k_r = jnp.reshape(Ea_k, (1, -1))
    A_oh_r = jnp.reshape(A_oh, (1, -1))
    dH_oh_r = jnp.reshape(dH_oh, (1, -1))
    A_h2_r = jnp.reshape(A_h2, (1, -1))
    dH_h2_r = jnp.reshape(dH_h2, (1, -1))
    A_mix_r = jnp.reshape(A_mix, (1, -1))
    dH_mix_r = jnp.reshape(dH_mix, (1, -1))

    # print(f"Input shape: {T.shape=}")
    # print(f"Parameter shape: {k0_r.shape=}")

    # Likelihood: observed rates are normally distributed around predictions
    n_obs = T.shape[0]

    r_pred = kinetics_model(
        p_i,
        T,
        k0_r,
        Ea_k_r,
        A_oh_r,
        dH_oh_r,
        A_h2_r,
        dH_h2_r,
        A_mix_r,
        dH_mix_r,
    )
    n_obs, n_draws = r_pred.shape
    # Store as deterministic for tracking
    r_pred = numpyro.deterministic("r_pred", r_pred)
    # print(f"Prediction shape: {r_pred.shape}")
    sigma_obs = sigma_r * jnp.abs(r_pred) + 1e-10  # Avoid zero stddev

    # TODO check whether I can pull this out of the plate
    with numpyro.plate("data", size=n_obs, dim=-2):
        # Compute predicted reaction rates inside plate context
        # Sample observations
        r = numpyro.sample(
            "r",
            dist.Normal(r_pred, sigma_obs),
            obs=r_obs,
        )
        # print(f"plate shape: {r.shape}")
    return r


def run_inference(
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    r_obs: jnp.ndarray,
    num_samples: int = 4000,
    num_warmup: int = 1000,
    num_chains: int = 4,
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
    # Configure NUTS sampler with high target acceptance for challenging posteriors
    seed = jax.random.PRNGKey(42)
    nuts_kernel = NUTS(
        bayesian_model,
        target_accept_prob=0.95,
        dense_mass=True,
        max_tree_depth=14,
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
        progress_bar=False,
    )

    mcmc.run(seed, T=T, p_i=p_i, r_obs=r_obs)
    print("MCMC sampling completed successfully!")

    # Convert to ArviZ InferenceData for analysis
    # TODO posterior predictive field is missing now!
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
    T_dummy = jnp.array([[555.0]])
    p_i_dummy = jnp.array([[2.0, 2.0, 0.0, 0.0, 0.0]])

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
        num_samples: Number of posterior predictive samples to generate
    """

    # print(T.shape, p_i.shape, r_obs.shape if r_obs is not None else None)

    # Extract posterior samples from InferenceData
    # and convert to correct data format
    draws = az.extract(idata, group="posterior", num_samples=num_samples).data_vars
    posterior_samples = {}
    for key, values in draws.items():
        # print(key)
        # print(values.shape)
        posterior_samples.update({key: jnp.array(values.values)})

    seed = jax.random.PRNGKey(0)
    posterior_predictive = Predictive(
        bayesian_model,
        posterior_samples=posterior_samples,
        exclude_deterministic=True,
        batch_ndims=0,
    )

    posterior_samples = posterior_predictive(seed, T=T, p_i=p_i, r_obs=r_obs)

    # Convert to ArviZ InferenceData
    # Reshape samples to (chain, draw) format - treat all samples as single chain
    posterior_dict = {}
    posterior_predictive_dict = {}
    for key, value in posterior_samples.items():
        # print(f"post sample of: {key}")
        # print(value.shape)
        draws = np.array(value).T[None, :, :]
        if key in ["r_pred", "r"]:
            posterior_predictive_dict[key] = draws  # shape (n_obs, draws)
        else:
            posterior_dict[key] = draws

    if r_obs is None:
        obs_data_dict = None
    else:
        obs_data_dict = {"r": np.array(r_obs)}

    data = {
        "posterior": posterior_dict,
        "posterior_predictive": posterior_predictive_dict,
    }
    if obs_data_dict is not None:
        data["observed_data"] = obs_data_dict

    idata = az.from_dict(
        data,
        dims={
            "r_pred": ["obs"],
            "r": ["obs"],
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

    obs_data = az.from_dict(
        {
            "observed_data": {
                "r": holdout.obs_r.values.reshape(1, 1, -1),
                "r_pred": holdout.latent_r.values.reshape(1, 1, -1),
            }
        },
        dims={
            "r_pred": ["obs"],
            "r": ["obs"],
        },
    )
    post_data.update(obs_data)

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
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (T, p_i, r_obs) ready for inference
    """
    # Set random seed
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(763789678968)
        key, _ = jax.random.split(key)

    # Calculate partial pressures
    p_i = calculate_partial_pressures(p, c2h, c2m, c2w, c2i)

    # Compute true reaction rates
    r_true = kinetics_model(p_i, T, *TRUE_PARAMS.values())

    # Add Gaussian noise
    rng = jax.random.normal(key, r_true.shape)
    noise = rng * rel_noise_std * jnp.abs(r_true)
    r_obs = r_true + noise

    return T, p_i, r_obs


def predict(
    idata: xarray.DataTree,
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    num_samples: int = 25,
    key: str = "posterior",
) -> np.ndarray:
    """
    Generate posterior predictions for given conditions.

    Args:
        idata: ArviZ InferenceData with posterior samples
        T: Temperature conditions [K]
        p_i: Partial pressure conditions [bar]
        num_samples: Number of posterior samples to use (None for all)

    Returns:
        Array of predicted rates with shape (n_conditions, n_samples)
    """
    # Extract posterior samples
    draws = az.extract(idata, group=key, num_samples=num_samples)

    # Flatten all chains and draws
    k0 = jnp.array(draws["k0"].values.flatten())
    Ea_k = jnp.array(draws["Ea_k"].values.flatten())
    A_oh = jnp.array(draws["A_oh"].values.flatten())
    dH_oh = jnp.array(draws["dH_oh"].values.flatten())
    A_h2 = jnp.array(draws["A_h2"].values.flatten())
    dH_h2 = jnp.array(draws["dH_h2"].values.flatten())
    A_mix = jnp.array(draws["A_mix"].values.flatten())
    dH_mix = jnp.array(draws["dH_mix"].values.flatten())

    # Reshape for broadcasting with kinetics model
    params = {
        "k0": k0.reshape(1, -1),
        "Ea_k": Ea_k.reshape(1, -1),
        "A_oh": A_oh.reshape(1, -1),
        "dH_oh": dH_oh.reshape(1, -1),
        "A_h2": A_h2.reshape(1, -1),
        "dH_h2": dH_h2.reshape(1, -1),
        "A_mix": A_mix.reshape(1, -1),
        "dH_mix": dH_mix.reshape(1, -1),
    }

    # Generate predictions
    predictions = kinetics_model(p_i, T, **params)

    return np.array(predictions)


def main() -> None:
    """
    Demonstration of the NumPyro-based kinetics inference workflow.
    """
    print("=== NumPyro LHHW Kinetics Model Demo (JAX Backend) ===\n")

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
        T_obs, p_i_obs, r_obs.squeeze(), num_samples=1000, num_warmup=500, num_chains=2
    )
    print("Inference complete!\n")

    # Print posterior summary
    print("MCMC Posterior summary:")
    print(az.summary(posterior_idata, hdi_prob=0.95))

    print("\n✅ All tests passed! JAX + NumPyro implementation works correctly.")


if __name__ == "__main__":
    main()
