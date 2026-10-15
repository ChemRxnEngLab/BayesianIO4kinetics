"""
Langmuir-Hinshelwood-Hougen-Watson (LHHW) Kinetics Model with Type II Deactivation

NumPyro
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
    "k_deact": jnp.array(2.8e-5),  # Deactivation rate constant [1/s]
    "EA_deact": jnp.array(35.0e3),  # Deactivation activation energy [J/mol]
    "n_deact": jnp.array(5),  # Deactivation order [-]
    "n_deact_p": jnp.array(1.5),  # Deactivation order [-]
}


def deact_ODE(t, y, params):
    """
    ODE for catalyst deactivation.

    Args:
        t: Time [s]
        y: Activity [-]
        params: Tuple of (k_deact, Ea_deact, n_deact, T, p_i)

    Returns:
        dy/dt: Rate of activity change
    """
    # print(y)
    k_deact, Ea_deact, n_deact, n_deact_p, T, p_i = params
    R_const = 8.314  # Gas constant [J/(mol·K)]
    T_ref = 550.0  # Reference temperature [K]

    k = jnp.expand_dims(
        k_deact * jnp.exp(Ea_deact / R_const * (1.0 / T_ref - 1.0 / T)), -2
    )
    p_m = jnp.expand_dims(p_i[:, 3:4], -2)
    dy = jnp.where(
        y <= 1e-3,
        0.0,
        -k * y**n_deact * p_m**n_deact_p,
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
    n_deact_p: jnp.ndarray,
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
    stepsize_controller = dfx.PIDController(
        # "optimized" values
        rtol=1e-7,
        atol=1e-9,
        pcoeff=0.3,
        icoeff=0.3,
        dcoeff=0.0,
        # old default values:
        # rtol=1e-8,
        # atol=1e-12,
    )

    a0 = jnp.ones((n_obs, 1, n_draws))

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        ode_term,
        solver,
        t0=0.0,
        t1=12.0 * 60.0 * 60.0,
        dt0=None,  # Let Diffrax choose initial step size
        y0=a0,
        args=(k_deact, Ea_deact, n_deact, n_deact_p, T, p_i),
        saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 12.0 * 60.0 * 60.0, 10)),
        stepsize_controller=stepsize_controller,
        # max_steps=2**18,
        throw=False,
    )

    a = solution.ys

    def on_fail():
        # jax.debug.print("Deactivation ODE solver failed.")
        # jax.debug.print("k_deact: {k}", k=k_deact)
        # jax.debug.print("Ea_deact: {E}", E=Ea_deact)
        # jax.debug.print("n_deact: {n}", n=n_deact)
        return jnp.full((10, n_obs, 1, n_draws), jnp.nan)

    def on_success():
        return a * r

    return jax.lax.cond(
        solution.result == dfx.RESULTS.max_steps_reached,
        on_fail,
        on_success,
    )

    # if solution.result == dfx.RESULTS.max_steps_reached:
    #     jax.debug.print("Deactivation ODE solver failed.")
    #     jax.debug.print("k_deact: {k}", k=k_deact)
    #     jax.debug.print("Ea_deact: {E}", E=Ea_deact)
    #     jax.debug.print("n_deact: {n}", n=n_deact)
    #     return jnp.full(r.shape, jnp.nan)

    # return a * r


def kinetics_model(
    p_i: jnp.ndarray,
    T: jnp.ndarray,
    k0: jnp.ndarray,
    Ea_k: jnp.ndarray,
    k_deact: jnp.ndarray,
    Ea_deact: jnp.ndarray,
    n_deact: jnp.ndarray,
    n_deact_p: jnp.ndarray,
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
    return _kinetics_core(p_i, T, k0, Ea_k, k_deact, Ea_deact, n_deact, n_deact_p)


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
    pass


if __name__ == "__main__":
    main()
