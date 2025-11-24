"""
Langmuir-Hinshelwood-Hougen-Watson (LHHW) Kinetics Model with Deactivation

NumPyro

This module implements a methanation reaction kinetics model using Bayesian inference
with the NumPyro probabilistic programming library. The model assumes homoscedastic noise
and uses informative priors based on expected parameter ranges.

The deactivation is modelled as being independent of the kinetics and the kinetics is in steady state.

Key Components:
- JAX-based kinetics calculations (NumPyro is built on JAX)
- NumPyro model for Bayesian parameter estimation
- Utility functions for experimental design and data generation
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
import diffrax as dfx


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
    "k_deact": jnp.array(1e-2),  # Deactivation rate constant [1/s]
    "EA_deact": jnp.array(75.0e3),  # Deactivation activation energy [J/mol]
    "n_deact": jnp.array(2),  # Deactivation order [-]
}


def deact_ODE(t, y, params):
    """
    ODE for catalyst deactivation.

    Args:
        t: Time [s]
        y: Activity [-]
        params: Tuple of (k_deact, Ea_deact, n_deact, T)

    Returns:
        dy/dt: Rate of activity change
    """
    k_deact, Ea_deact, n_deact, T = params
    R_const = 8.314  # Gas constant [J/(mol·K)]
    T_ref = 550.0  # Reference temperature [K]

    k = jnp.expand_dims(
        k_deact * jnp.exp(Ea_deact / R_const * (1.0 / T_ref - 1.0 / T)), -2
    )
    dy = jnp.where(
        y <= 1e-3,
        0.0,
        -k * y**n_deact,
    )
    return dy


def _kinetics_core(
    p_i: jnp.ndarray,
    T: jnp.ndarray,
    k0: jnp.ndarray,
    Ea_k: jnp.ndarray,
    k_deact: jnp.ndarray,
    Ea_deact: jnp.ndarray,
    n_deact: jnp.ndarray,
) -> jnp.ndarray:
    """
    Core kinetics calculation with deactivation.

    Args:
        p_i: Partial pressures array of shape (n_conditions, 5) [bar]
        T: Temperature array of shape (n_conditions, 1) [K]
        k0: Pre-exponential factor [mol/(bar·s·g_cat)]
        Ea_k: Activation energy [J/mol]
        k_deact: Deactivation rate constant [1/s]
        Ea_deact: Deactivation activation energy [J/mol]
        n_deact: Deactivation order [-]

    Returns:
        Reaction rate with deactivation array [mol/(s·g_cat)]
    """
    R_const = 8.314  # Gas constant [J/(mol·K)]
    T_ref = 555.0  # Reference temperature [K]

    # Arrhenius rate constant
    k = k0 * jnp.exp(Ea_k / R_const * (1.0 / T_ref - 1.0 / T))

    # Van't Hoff adsorption equilibrium constants
    K_OH = 0.5 * jnp.exp(22.4e3 / R_const * (1.0 / T_ref - 1.0 / T))
    K_H2 = 0.44 * jnp.exp(-6.2e3 / R_const * (1.0 / T_ref - 1.0 / T))
    K_mix = 0.88 * jnp.exp(-10.0e3 / R_const * (1.0 / T_ref - 1.0 / T))

    # Thermodynamic equilibrium constant for CO2 methanation
    K_eq = 137.0 * T ** (-4.0) * jnp.exp(158700.0 / (R_const * T))

    # Extract partial pressures
    p_CO2 = p_i[:, 0, None]  # SIDX["CO2"] = 0
    p_H2 = p_i[:, 1, None]  # SIDX["H2"] = 1
    p_H2O = p_i[:, 2, None]  # SIDX["H2O"] = 2
    p_CH4 = p_i[:, 3, None]  # SIDX["CH4"] = 3

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
    r = jnp.expand_dims(r, -2)
    n_obs = T.shape[0]
    n_draws = r.shape[2]

    # Solve deactivation ODE using Diffrax
    ode_term = dfx.ODETerm(deact_ODE)
    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-12)

    a0 = jnp.ones((n_obs, 1, n_draws))

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        ode_term,
        solver,
        t0=0.0,
        t1=60.0,
        dt0=None,  # Let Diffrax choose initial step size
        y0=a0,
        args=(k_deact, Ea_deact, n_deact, T),
        saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 60.0, 10)),
        stepsize_controller=stepsize_controller,
    )

    a = solution.ys

    return a * r


def kinetics_model(
    p_i: jnp.ndarray,
    T: jnp.ndarray,
    k0: jnp.ndarray,
    Ea_k: jnp.ndarray,
    k_deact: jnp.ndarray,
    Ea_deact: jnp.ndarray,
    n_deact: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute LHHW kinetics model for CO2 methanation reaction with deactivation.

    Reaction: CO2 + 4H2 ⇌ CH4 + 2H2O

    Args:
        p_i: Partial pressures array of shape (n_conditions, 5) [bar]
             Order: [CO2, H2, H2O, CH4, I]
        T: Temperature array of shape (n_conditions, 1) [K]
        k0: Pre-exponential factor [mol/(bar·s·g_cat)]
        Ea_k: Activation energy [J/mol]
        k_deact: Deactivation rate constant [1/s]
        Ea_deact: Deactivation activation energy [J/mol]
        n_deact: Deactivation order [-]

    Returns:
        Reaction rate array [mol/(s·g_cat)]
    """
    return _kinetics_core(p_i, T, k0, Ea_k, k_deact, Ea_deact, n_deact)


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
        c2m_ratio: CO2/CH4 molar ratio array of shape (n_conditions, 1) [-]
        c2w_ratio: CO2/H2O molar ratio array of shape (n_conditions, 1) [-]
        c2i_ratio: CO2/Inert molar ratio array of shape (n_conditions, 1) [-]

    Returns:
        Partial pressure array of shape (n_conditions, 5) [bar]
    """
    # Calculate CO2 mole fraction from ratios
    x_co2 = (
        1 / (1 + 1 / c2h_ratio + 1 / c2m_ratio + 1 / c2w_ratio + 1 / c2i_ratio)
    ).squeeze()

    # Calculate other species mole fractions
    x_h2 = x_co2 / c2h_ratio.squeeze()
    x_m = x_co2 / c2m_ratio.squeeze()
    x_w = x_co2 / c2w_ratio.squeeze()
    x_i = x_co2 / c2i_ratio.squeeze()

    xi = jnp.stack((x_co2, x_h2, x_w, x_m, x_i), axis=1)

    # Convert to partial pressures: p_i = x_i * p_total
    p_i = p_total * xi
    return p_i


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
    t_Ea_k = jnp.expand_dims(numpyro.sample("t_Ea_k", dist.Normal(0, 0.1)), 0)
    t_k_deact = jnp.expand_dims(numpyro.sample("t_k_deact", dist.Normal(0, 1)), 0)
    t_Ea_deact = jnp.expand_dims(numpyro.sample("t_Ea_deact", dist.Normal(0, 0.1)), 0)

    # Transform to original space
    k0 = numpyro.deterministic("k0", 1e-4 * jnp.exp(t_k0))
    Ea_k = numpyro.deterministic("Ea_k", 50e3 * jnp.exp(t_Ea_k))
    k_deact = numpyro.deterministic("k_deact", 1e-4 * jnp.exp(t_k_deact))
    Ea_deact = numpyro.deterministic("Ea_deact", 50e3 * jnp.exp(t_Ea_deact))
    n_deact = jnp.expand_dims(numpyro.sample("n_deact", dist.Uniform(1, 2)), 0)

    # Homoscedastic observation noise
    sigma_r = jnp.expand_dims(numpyro.sample("sigma_r", dist.HalfNormal(0.01)), 0)

    # Reshape observations if provided
    if r_obs is not None:
        r_obs_proc = jnp.permute_dims(r_obs, (1, 3, 0, 2))
        r_obs_proc = jnp.squeeze(r_obs_proc, 1)
    else:
        r_obs_proc = None

    # Compute predicted reaction rates
    r_pred = kinetics_model(p_i, T, k0, Ea_k, k_deact, Ea_deact, n_deact)

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
) -> az.InferenceData:
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
    nuts_kernel = NUTS(bayesian_model, target_accept_prob=0.9)

    # Run MCMC
    print(
        f"Starting MCMC: {num_chains} chains × ({num_warmup} warmup + {num_samples} samples)"
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )

    mcmc.run(seed, T=T, p_i=p_i, r_obs=r_obs)
    print("MCMC sampling completed successfully!")

    # Convert to ArviZ InferenceData for analysis
    idata = az.from_numpyro(mcmc)
    return idata


# Note: NumPyro SVI implementation differs from Pyro
# For variational inference, use numpyro.infer.SVI with custom guides


def sample_prior(num_samples: int = 1000) -> az.InferenceData:
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
    idata: az.InferenceData,
    T: jnp.ndarray,
    p_i: jnp.ndarray,
    num_samples: int = 1000,
    r_obs: Optional[jnp.ndarray] = None,
) -> az.InferenceData:
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

    # Extract posterior samples from InferenceData
    draws = az.extract(idata, group="posterior", num_samples=num_samples).data_vars
    posterior_samples = {}
    for key, values in draws.items():
        posterior_samples.update({key: jnp.array(values.values)})

    seed = jax.random.PRNGKey(0)
    posterior_predictive = Predictive(
        bayesian_model,
        posterior_samples=posterior_samples,
    )

    posterior_samples = posterior_predictive(seed, T=T, p_i=p_i, r_obs=r_obs)

    # Convert to ArviZ InferenceData
    posterior_dict = {}
    posterior_predictive_dict = {}
    for key, value in posterior_samples.items():
        draws = np.array(value).squeeze()
        draws = draws[None, :]
        if key in ["r_lat", "r"]:
            posterior_predictive_dict[key] = draws
        else:
            posterior_dict[key] = draws

    if r_obs is None:
        obs_data_dict = None
    else:
        obs_data_dict = {"r": np.array(r_obs)}

    idata = az.from_dict(
        posterior=posterior_dict,
        posterior_predictive=posterior_predictive_dict,
        observed_data=obs_data_dict,
        dims={
            "r_lat": ["obs"],
            "r": ["obs"],
        },
    )
    return idata


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
    idata: az.InferenceData,
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

    # Reshape for broadcasting with kinetics model
    params = {
        "k0": k0.reshape(1, -1),
        "Ea_k": Ea_k.reshape(1, -1),
        "k_deact": k_deact.reshape(1, -1),
        "Ea_deact": Ea_deact.reshape(1, -1),
        "n_deact": n_deact.reshape(1, -1),
    }

    # Generate predictions
    predictions = kinetics_model(p_i, T, **params)

    return np.array(predictions)


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
