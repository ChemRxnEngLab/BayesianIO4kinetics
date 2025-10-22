import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import ICIW_Plots
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from ICIW_Plots import make_square_subplots, cm2inch
from HSA_Cam_B_001_lib import forward_solve

plt.style.use("ICIWstyle")


def bmodel(y0_obs, t_obs, params_obs, y_obs):
    y0 = pt.as_tensor_variable(y0_obs)
    t = pt.as_tensor_variable(t_obs)
    indep_params = pt.as_tensor_variable(params_obs)
    with pm.Model() as model:
        # Priors for unknown model parameters
        k_deact = pm.Lognormal("k_deact", mu=np.log(1e-4), sigma=1)
        EA_deact = pm.Normal("EA_deact", mu=60000, sigma=5000)
        n_deact = pm.Uniform("n_deact", lower=0, upper=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        y_hat = pm.Deterministic(
            "y_hat",
            forward_solve(
                y0,
                t,
                pt.stack((*indep_params, k_deact, EA_deact, n_deact)),
            ),
        )
        # just observe the H2 consumption

        LH2 = pm.Normal("LH2", mu=y_hat[:, 2], sigma=sigma, observed=y_obs[:, 0])

    return model
