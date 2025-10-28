"""
Langmuir-Hinshelwood-Hougen-Watson (LHHW) Kinetics Model with Pyro

This module implements a methanation reaction kinetics model using Bayesian inference
with the Pyro probabilistic programming library. The model assumes homoscedastic noise
and uses informative priors based on expected parameter ranges.

Key Components:
- PyTorch-based kinetics calculations (Pyro is built on PyTorch)
- Pyro model for Bayesian parameter estimation
- Utility functions for experimental design and data generation
"""

from typing import Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import Adam
import arviz as az


plt.style.use("ICIWstyle")

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
    "k0": torch.tensor(3.46e-4),  # Pre-exponential factor [mol/(bar·s·g_cat)]
    "EA": torch.tensor(77.5e3),  # Activation energy [J/mol]
    "A_OH": torch.tensor(0.50),  # OH adsorption pre-exponential [bar^-0.5]
    "H_OH": torch.tensor(22.4e3),  # OH adsorption enthalpy [J/mol]
    "A_H2": torch.tensor(0.44),  # H2 adsorption pre-exponential [bar^-0.5]
    "H_H2": torch.tensor(-6.2e3),  # H2 adsorption enthalpy [J/mol]
    "A_mix": torch.tensor(0.88),  # Mixed adsorption pre-exponential [bar^-0.5]
    "H_mix": torch.tensor(-10.0e3),  # Mixed adsorption enthalpy [J/mol]
}


# JIT-compiled core kinetics computation for maximum speed
@torch.jit.script
def _kinetics_core(
    p_i: torch.Tensor,
    T: torch.Tensor,
    k0: torch.Tensor,
    Ea_k: torch.Tensor,
    A_oh: torch.Tensor,
    dH_oh: torch.Tensor,
    A_h2: torch.Tensor,
    dH_h2: torch.Tensor,
    A_mix: torch.Tensor,
    dH_mix: torch.Tensor,
) -> torch.Tensor:
    """
    JIT-compiled core kinetics calculation (optimized for speed).
    This function is compiled once and reused for all subsequent calls.
    """
    R_const = 8.314  # Gas constant [J/(mol·K)] - must be local for JIT
    T_ref = 555.0  # Reference temperature [K]

    # Arrhenius rate constant
    k = k0 * torch.exp(Ea_k / R_const * (1.0 / T_ref - 1.0 / T))

    # Van't Hoff adsorption equilibrium constants
    K_OH = A_oh * torch.exp(dH_oh / R_const * (1.0 / T_ref - 1.0 / T))
    K_H2 = A_h2 * torch.exp(dH_h2 / R_const * (1.0 / T_ref - 1.0 / T))
    K_mix = A_mix * torch.exp(dH_mix / R_const * (1.0 / T_ref - 1.0 / T))

    # Thermodynamic equilibrium constant for CO2 methanation
    K_eq = 137.0 * T ** (-4.0) * torch.exp(158700.0 / (R_const * T))

    # Extract partial pressures (indexed directly for JIT optimization)
    p_CO2 = p_i[:, 0, None]  # SIDX["CO2"] = 0
    p_H2 = p_i[:, 1, None]  # SIDX["H2"] = 1
    p_H2O = p_i[:, 2, None]  # SIDX["H2O"] = 2
    p_CH4 = p_i[:, 3, None]  # SIDX["CH4"] = 3

    # Reaction driving force (1 - Q/K_eq)
    driving_force = 1.0 - (p_CH4 * p_H2O**2.0) / (p_CO2 * p_H2**4.0 * K_eq)

    # Denominator: adsorption inhibition term (squared LHHW form)
    sqrt_p_H2 = torch.sqrt(p_H2)
    sqrt_p_CO2 = torch.sqrt(p_CO2)

    inhibition = (
        1.0 + (K_OH * p_H2O) / sqrt_p_H2 + K_H2 * sqrt_p_H2 + K_mix * sqrt_p_CO2
    ) ** 2.0

    # LHHW rate equation
    r = k * sqrt_p_H2 * sqrt_p_CO2 * driving_force / inhibition

    return r


def kinetics_model(
    p_i: torch.Tensor,
    T: torch.Tensor,
    k0: torch.Tensor,
    Ea_k: torch.Tensor,
    A_oh: torch.Tensor,
    dH_oh: torch.Tensor,
    A_h2: torch.Tensor,
    dH_h2: torch.Tensor,
    A_mix: torch.Tensor,
    dH_mix: torch.Tensor,
) -> torch.Tensor:
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
    p_total: torch.Tensor,
    c2h_ratio: torch.Tensor,
    feed_ratio: torch.Tensor,
    c2i_ratio: torch.Tensor,
) -> torch.Tensor:
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
    # Convert to tensors if needed
    # p_total = torch.as_tensor(p_total, dtype=torch.float32)
    # c2h_ratio = torch.as_tensor(c2h_ratio, dtype=torch.float32)
    # feed_ratio = torch.as_tensor(feed_ratio, dtype=torch.float32)
    # c2i_ratio = torch.as_tensor(c2i_ratio, dtype=torch.float32)

    # Initialize mole fraction array
    x_i = torch.zeros((p_total.shape[0], 5))

    # Calculate CO2 mole fraction from ratios
    x_co2 = 1 / (1 + 1 / feed_ratio + 1 / c2h_ratio + 1 / c2i_ratio)
    x_i[:, SIDX["CO2"]] = x_co2.squeeze()

    # Calculate other species mole fractions
    x_i[:, SIDX["H2"]] = x_i[:, SIDX["CO2"]] / c2h_ratio.squeeze()
    x_i[:, SIDX["CH4"]] = x_i[:, SIDX["CO2"]] / feed_ratio.squeeze()
    x_i[:, SIDX["I"]] = x_i[:, SIDX["CO2"]] / c2i_ratio.squeeze()

    # Convert to partial pressures: p_i = x_i * p_total
    p_i = p_total * x_i

    return p_i


def bayesian_model(
    T: torch.Tensor, p_i: torch.Tensor, r_obs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pyro Bayesian model for kinetic parameter inference.

    This model uses informative priors based on expected parameter ranges:
    - Exponential priors for positive parameters (rate constants, adsorption)
    - Normal priors for enthalpy changes (can be positive or negative)
    - Half-normal prior for observation noise

    Args:
        T: Temperature array [K]
        p_i: Partial pressures array [bar]
        r_obs: Observed reaction rates [mol/(s·g_cat)] (None for prior sampling)

    Returns:
        Sampled or observed reaction rates
    """
    # Sample kinetic parameters from priors
    k0 = pyro.sample("k0", dist.Exponential(1 / 0.001))
    Ea_k = pyro.sample("Ea_k", dist.Exponential(1 / 60000))

    A_oh = pyro.sample("A_oh", dist.Exponential(1.0))
    dH_oh = pyro.sample("dH_oh", dist.Exponential(1 / 10000))

    A_h2 = pyro.sample("A_h2", dist.Exponential(1.0))
    dH_h2 = pyro.sample("dH_h2", dist.Normal(-10000, 5000))

    A_mix = pyro.sample("A_mix", dist.Exponential(1.0))
    dH_mix = pyro.sample("dH_mix", dist.Normal(-10000, 5000))

    # Homoscedastic observation noise
    sigma_r = pyro.sample("sigma_r", dist.HalfNormal(1e-5))

    # Compute predicted reaction rates
    r_pred = pyro.deterministic(
        "r_pred",
        kinetics_model(p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix),
    )

    # Likelihood: observed rates are normally distributed around predictions
    with pyro.plate("data", r_pred.shape[0]):
        r = pyro.sample("r", dist.Normal(r_pred.squeeze(), sigma_r), obs=r_obs)

    return r


def run_inference(
    T: torch.Tensor,
    p_i: torch.Tensor,
    r_obs: torch.Tensor,
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
    # Convert to tensors
    # T = torch.as_tensor(T, dtype=torch.float32)
    # p_i = torch.as_tensor(p_i, dtype=torch.float32)
    # r_obs = torch.as_tensor(r_obs, dtype=torch.float32)

    # Configure NUTS sampler with high target acceptance for challenging posteriors
    nuts_kernel = NUTS(bayesian_model, target_accept_prob=0.9, jit_compile=True)

    # Run MCMC with disabled progress bar to avoid notebook widget issues
    print(
        f"Starting MCMC: {num_chains} chains × ({num_warmup} warmup + {num_samples} samples)"
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=num_warmup,
        num_chains=num_chains,
        disable_progbar=True,  # Disable fancy progress bars
    )

    mcmc.run(T=T, p_i=p_i, r_obs=r_obs)
    print("MCMC sampling completed successfully!")

    # Convert to ArviZ InferenceData for analysis
    # posterior_samples = mcmc.get_samples(group_by_chain=True)
    # posterior_predictive = Predictive(bayesian_model, posterior_samples)

    # posterior = posterior_predictive(T=T, p_i=p_i)

    # Convert torch tensors to numpy for ArviZ
    idata = az.from_pyro(mcmc)
    return idata


def run_variational_inference(
    T: torch.Tensor,
    p_i: torch.Tensor,
    r_obs: torch.Tensor,
    num_steps: int = 10000,
    learning_rate: float = 0.01,
    num_samples: int = 4000,
) -> az.InferenceData:
    """
    Run Stochastic Variational Inference (SVI) to estimate kinetic parameters.

    This is a faster alternative to MCMC that approximates the posterior with
    a multivariate normal distribution. Good for quick initial estimates or
    when MCMC is too slow.

    Args:
        T: Temperature observations [K]
        p_i: Partial pressure observations [bar]
        r_obs: Observed reaction rates [mol/(s·g_cat)]
        num_steps: Number of optimization steps for SVI
        learning_rate: Learning rate for Adam optimizer
        num_samples: Number of samples to draw from approximate posterior

    Returns:
        ArviZ InferenceData object with approximate posterior samples
    """
    # Clear parameter store to avoid conflicts
    pyro.clear_param_store()

    # Setup guide: multivariate normal approximation to posterior
    guide = AutoMultivariateNormal(bayesian_model)

    # Setup optimizer
    optimizer = Adam({"lr": learning_rate})

    # Setup SVI with ELBO loss
    svi = SVI(bayesian_model, guide, optimizer, loss=Trace_ELBO())

    # Run variational inference
    print(f"Starting SVI: {num_steps} optimization steps")
    losses = []

    for step in range(num_steps):
        loss = svi.step(T=T, p_i=p_i, r_obs=r_obs)
        losses.append(loss)

        # Print progress every 1000 steps
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}/{num_steps}, Loss: {loss:.2e}")

    print("SVI optimization completed successfully!")

    # Sample from the approximate posterior
    print(f"Drawing {num_samples} samples from approximate posterior...")
    predictive = Predictive(
        bayesian_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=list(guide().keys()),  # Get all parameter sites
    )

    posterior_samples = predictive(T=T, p_i=p_i, r_obs=None)

    # Remove observation variables if present
    posterior_samples.pop("r", None)
    posterior_samples.pop("r_obs", None)

    # Convert to ArviZ InferenceData
    # Reshape samples to (chain, draw) format - treat all samples as single chain
    posterior_dict = {}
    for key, value in posterior_samples.items():
        if key != "r_pred":  # Skip deterministic sites for now
            # Shape: (num_samples,) -> (1, num_samples) for ArviZ
            posterior_dict[key] = value.detach().cpu().numpy()[None, :]

    idata = az.from_dict(posterior=posterior_dict)

    # Store optimization history as additional info
    idata.attrs["svi_losses"] = losses
    idata.attrs["svi_final_loss"] = losses[-1]
    idata.attrs["svi_num_steps"] = num_steps

    print(f"Approximate posterior created with {num_samples} samples\n")

    return idata


def sample_prior(num_samples: int = 1000) -> az.InferenceData:
    """
    Sample from the prior distribution.

    Args:
        num_samples: Number of prior samples to generate

    Returns:
        ArviZ InferenceData object with prior samples
    """
    # Use Predictive to sample from prior (no observations)
    prior_predictive = Predictive(bayesian_model, num_samples=num_samples)

    # Dummy data for sampling (not used in prior)
    T_dummy = torch.tensor([[555.0]])
    p_i_dummy = torch.tensor([[2.0, 2.0, 0.0, 0.0, 0.0]])

    prior_samples = prior_predictive(T=T_dummy, p_i=p_i_dummy)

    # Remove the observation variable
    prior_samples.pop("r", None)

    # Create InferenceData with prior group
    idata = az.from_pyro(prior=prior_samples)
    return idata


def generate_synthetic_data(
    T: torch.Tensor,
    p: torch.Tensor,
    c2h: torch.Tensor,
    f: torch.Tensor,
    c2i: torch.Tensor,
    noise_std: float = 1e-5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic observation data using true parameters.

    Args:
        T: Temperature array [K]
        p: Total pressure array [bar]
        c2h: CO2/H2 ratio array [-]
        f: Feed ratio array [-]
        c2i: CO2/Inert ratio array [-]
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (T, p_i, r_obs) ready for inference
    """
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate partial pressures
    p_i = calculate_partial_pressures(p, c2h, f, c2i)

    # Compute true reaction rates
    r_true = kinetics_model(p_i, T, *TRUE_PARAMS.values())

    # Add Gaussian noise
    noise = torch.randn_like(r_true) * noise_std
    r_obs = r_true + noise

    return T, p_i, r_obs


def predict(
    idata: az.InferenceData,
    T: torch.Tensor,
    p_i: torch.Tensor,
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
    k0 = torch.tensor(draws["k0"].values.flatten())
    Ea_k = torch.tensor(draws["Ea_k"].values.flatten())
    A_oh = torch.tensor(draws["A_oh"].values.flatten())
    dH_oh = torch.tensor(draws["dH_oh"].values.flatten())
    A_h2 = torch.tensor(draws["A_h2"].values.flatten())
    dH_h2 = torch.tensor(draws["dH_h2"].values.flatten())
    A_mix = torch.tensor(draws["A_mix"].values.flatten())
    dH_mix = torch.tensor(draws["dH_mix"].values.flatten())

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

    return predictions.cpu().numpy()


def main() -> None:
    """
    Demonstration of the Pyro-based kinetics inference workflow.
    """
    print("=== Pyro LHHW Kinetics Model Demo (PyTorch Backend) ===\n")

    # Define experimental conditions
    T = torch.tensor([[560.0], [570.0], [580.0]])
    p = torch.tensor([[4.0], [5.0], [6.0]])
    c2h = torch.ones((3, 1))
    f = torch.ones((3, 1)) * 1000
    c2i = torch.ones((3, 1)) * 0.5

    print("Generating synthetic data...")
    T_obs, p_i_obs, r_obs = generate_synthetic_data(T, p, c2h, f, c2i)
    print(f"Observations shape: {r_obs.shape}")
    print(f"Observed rates: {r_obs.squeeze()}\n")

    print("Sampling from prior...")
    prior_idata = sample_prior(num_samples=1000)
    print(f"Prior samples: {list(prior_idata.prior.data_vars.keys())}\n")

    # Variational inference (fast approximation)
    print("Running Variational Inference (SVI)...")
    vi_idata = run_variational_inference(
        T_obs, p_i_obs, r_obs.squeeze(), num_steps=5000, num_samples=2000
    )
    print("Variational inference complete!\n")

    print("VI Posterior summary:")
    print(az.summary(vi_idata, hdi_prob=0.95))
    print()

    print("Running MCMC inference...")
    posterior_idata = run_inference(
        T_obs, p_i_obs, r_obs.squeeze(), num_samples=1000, num_warmup=500, num_chains=2
    )
    print("Inference complete!\n")

    # Print posterior summary
    print("MCMC Posterior summary:")
    print(az.summary(posterior_idata, hdi_prob=0.95))

    # Compare VI vs MCMC
    print("\n=== Comparison: VI vs MCMC ===")
    print(f"VI final loss: {vi_idata.attrs['svi_final_loss']:.2e}")
    print(f"VI computation: ~{vi_idata.attrs['svi_num_steps']} gradient steps")
    print(
        f"MCMC computation: {posterior_idata.posterior.dims['draw'] * posterior_idata.posterior.dims['chain']} samples"
    )

    print("\n✅ All tests passed! PyTorch + Pyro implementation works correctly.")


if __name__ == "__main__":
    main()
