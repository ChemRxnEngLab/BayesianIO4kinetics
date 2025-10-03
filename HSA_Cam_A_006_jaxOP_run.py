# %%
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import ICIW_Plots
from scipy.integrate import odeint
import pytensor as pt
from HSA_Cam_A_006_jaxOp import (
    bmodel,
    bmodel_batch,
    plot_true_data,
    plot_observed_data,
    plot_inference,
    model_inference,
    theta_true,
    y0_true,
    sigma,
)
from HSA_Cam_A_006_lib_jaxOP import _forward_solve, _forward_solve_batch
import xarray

pt.config.mode = "JAX"
plt.style.use("ICIWstyle")


def my_callback(trace, draw):
    if len(trace) % 100 == 0:
        print(f"Draw {len(trace)}")


def main():
    times = np.linspace(0, 1 * 60 * 60, 100, dtype=np.float64)
    times_batch = np.tile(times, (6, 1))
    T_vals = np.array([280.0, 300.0, 320.0, 350.0, 380.0, 400.0], dtype=np.float64)
    theta = theta_true.copy()
    theta_batch = np.tile(theta, (6, 1))
    theta_batch[:, 1] = T_vals  # varying temperature
    y0_true_batch = np.tile(y0_true, (6, 1))

    y_true = xarray.DataArray(
        np.zeros((len(T_vals), len(times), 5)),
        dims=["temperature", "time", "state"],
        coords={
            "temperature": T_vals.tolist(),
            "time": times,
            "state": ["n_g_con", "c_Al", "c_Bl", "c_Cl", "c_Dl"],
        },
    )
    y_true[:, :, :] = _forward_solve_batch(y0_true_batch, times_batch, theta_batch)
    y_true[:, :, 0] *= 1000  # convert m3 to L H2 used (up to a constant atleast)

    # Add noise
    t_obs = times_batch[:, ::10]
    y_obs = np.random.normal(y_true, sigma)[:, ::10, :]  # downsample and add noise
    y_obs = xarray.DataArray(
        y_obs,
        dims=["temperature", "time", "state"],
        coords={
            "temperature": T_vals.tolist(),
            "time": t_obs[0, :],
            "state": ["n_g_con", "c_Al", "c_Bl", "c_Cl", "c_Dl"],
        },
    )

    model_batch = bmodel_batch(t_obs, T_vals, y_obs.values)
    vars_list = list(model_batch.values_to_rvs.keys())[:-1]
    with model_batch:
        posterior_idata = pm.sample(
            # step=[pm.DEMetropolisZ(vars_list)],
            draws=1000,
            tune=100,
            # target_accept=0.9,
            random_seed=23489,
            return_inferencedata=True,
            # nuts_sampler="numpyro",
            cores=8,
            chains=8,
            progressbar=False,
            callbacks=[my_callback],
        )

    print(
        f"ESS[EA1]: {az.ess(posterior_idata, method='bulk', relative=False)['EA1'].values:.1f}"
    )
    print(
        f"ESS/s[EA1]: {az.ess(posterior_idata, method='bulk', relative=True)['EA1'].values:.3f}"
    )
    posterior_idata.to_netcdf("HSA_Cam_A_006_jaxOP_posterior.nc")


if __name__ == "__main__":
    main()
