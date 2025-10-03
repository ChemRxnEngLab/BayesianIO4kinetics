import os
import typing as t
import pytensor as pt
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import xarray
import numpyro
import ICIW_Plots
from lib.BIBOutil import _from_posterior
from HSA_Cam_A_006_lib_jaxOP import (
    forward_solve,
    forward_solve_batch,
    _forward_solve,
    _forward_solve_batch,
)

plt.style.use("ICIWstyle")
pt.config.mode = "JAX"
os.environ["MKL_THREADING_LAYER"] = "GNU"
pt.config.blas__ldflags = "-lmkl_rt"
numpyro.set_host_device_count(4)
# For reproducibility
np.random.seed(23489)

y0_true = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float64)
theta_true = np.array(
    [10e5, 300, 3e-3, 0.0001, 40000, 0.00001, 65000], dtype=np.float64
)
# sigma = 0
sigma = 0.5


def bmodel(t_obs, T_obs, y_obs, prior: t.Optional[az.InferenceData] = None):
    with pm.Model() as model:
        if prior is None:
            # Priors for unknown model parameters
            kla = pm.Lognormal("kla", mu=np.log(3e-3), sigma=1)
            A01 = pm.Lognormal("A01", mu=np.log(1e-4), sigma=1)
            EA1 = pm.Normal("EA1", mu=60000, sigma=5000)
            A02 = pm.Lognormal("A02", mu=np.log(1e-4), sigma=1)
            EA2 = pm.Normal("EA2", mu=60000, sigma=5000)
            sigma = pm.HalfNormal("sigma", sigma=1)
        else:
            kla = _from_posterior("kla", prior)
            A01 = _from_posterior("A01", prior)
            EA1 = _from_posterior("EA1", prior)
            A02 = _from_posterior("A02", prior)
            EA2 = _from_posterior("EA2", prior)
            sigma = _from_posterior("sigma", prior)

        y_hat = pm.Deterministic(
            "y_hat",
            forward_solve(
                pm.math.stack([*y0_true]),
                pm.math.stack([*t_obs]),
                pm.math.stack((theta_true[0], T_obs, kla, A01, EA1, A02, EA2)),
            ),
        )
        # just observe the H2 consumption

        LH2 = pm.Normal("LH2", mu=y_hat[:, 0] * 1000, sigma=sigma, observed=y_obs[:, 0])

    return model


def bmodel_batch(t_obs, T_obs, y_obs, prior: t.Optional[az.InferenceData] = None):
    y0_batch = pt.tensor.tile(y0_true, (t_obs.shape[0], 1))

    t_obs_batch = pt.tensor.as_tensor_variable(t_obs)
    p_batch = pt.tensor.tile(theta_true[0], (t_obs.shape[0],))
    indep_params = pt.tensor.stack(
        (
            p_batch,
            T_obs,
        ),
        axis=1,
    )

    with pm.Model() as model:
        if prior is None:
            # Priors for unknown model parameters
            kla = pm.Lognormal("kla", mu=np.log(3e-3), sigma=1)
            A01 = pm.Lognormal("A01", mu=np.log(1e-4), sigma=1)
            EA1 = pm.Normal("EA1", mu=60000, sigma=5000)
            A02 = pm.Lognormal("A02", mu=np.log(1e-4), sigma=1)
            EA2 = pm.Normal("EA2", mu=60000, sigma=5000)
            sigma = pm.HalfNormal("sigma", sigma=1)
        else:
            kla = _from_posterior("kla", prior)
            A01 = _from_posterior("A01", prior)
            EA1 = _from_posterior("EA1", prior)
            A02 = _from_posterior("A02", prior)
            EA2 = _from_posterior("EA2", prior)
            sigma = _from_posterior("sigma", prior)

        params = pt.tensor.tile(
            pt.tensor.stack((kla, A01, EA1, A02, EA2), axis=1),
            (t_obs.shape[0], 1),
        )

        theta_batch = pt.tensor.concatenate(
            (indep_params, params),
            axis=1,
        )

        y_hat = pm.Deterministic(
            "y_hat",
            forward_solve_batch(
                y0_batch,
                t_obs_batch,
                theta_batch,
            ),
        )
        # just observe the H2 consumption

        LH2 = pm.Normal(
            "LH2", mu=y_hat[:, :, 0] * 1000, sigma=sigma, observed=y_obs[:, :, 0]
        )

    return model


def plot_true_data(axs, times, y_true, lw=2, alpha=1):
    axs[0, 0].plot(
        times,
        y_true[:, 0],
        label="gas consumed",
        linestyle="dashed",
        color="red",
        lw=lw,
        alpha=alpha,
    )
    axs[0, 1].plot(
        times,
        y_true[:, 1],
        label="A(l)",
        linestyle="dashed",
        color="blue",
        lw=lw,
        alpha=alpha,
    )
    axs[0, 1].plot(
        times,
        y_true[:, 2],
        label="B(l)",
        linestyle="dashed",
        color="green",
        lw=lw,
        alpha=alpha,
    )
    axs[0, 1].plot(
        times,
        y_true[:, 3],
        label="C(l)",
        linestyle="dashed",
        color="black",
        lw=lw,
        alpha=alpha,
    )


def plot_observed_data(axs, t_obs, y_obs):
    axs[0, 0].scatter(
        t_obs,
        y_obs[:, 0],
        marker="o",
        color="red",
    )
    axs[0, 1].scatter(
        t_obs,
        y_obs[:, 1],
        marker="o",
        color="blue",
    )
    axs[0, 1].scatter(
        t_obs,
        y_obs[:, 2],
        marker="o",
        color="green",
    )
    axs[0, 1].scatter(
        t_obs,
        y_obs[:, 3],
        marker="o",
        color="black",
    )


def model_inference(
    trace,
    y0,
    t,
    theta_indep=None,
    n_samples=25,
    group="posterior",
):
    if theta_indep is None:
        theta_indep = (theta_true[0], theta_true[1])
    trace_df: xarray.Dataset = az.extract(
        trace,
        group=group,
        num_samples=n_samples,
        var_names=["kla", "A01", "EA1", "A02", "EA2"],
    )
    theta_dep = trace_df.to_array().T
    y0s = np.tile(y0, (n_samples, 1))
    ts = np.tile(t, (n_samples, 1))
    thetas = np.empty((n_samples, 7))
    thetas[:, :2] = theta_indep
    thetas[:, 2:] = theta_dep
    y_hat = np.array(_forward_solve_batch(y0s, ts, thetas))
    return y_hat


def plot_inference(
    ax,
    trace,
    times,
    group="prior",
    theta_indep=None,
    num_samples=25,
    **kwargs,
):
    y_hat = model_inference(
        trace,
        y0_true,
        times,
        theta_indep=theta_indep,
        n_samples=num_samples,
        group=group,
    )
    y_hat[:, :, 0] *= 1000  # convert m3 to L H2 used (up to a constant atleast)
    for _y_hat in y_hat:
        plot_true_data(ax, times, _y_hat, **kwargs)


if __name__ == "__main__":
    # Times for observation
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

    ## Plotting the base model
    fig = ICIW_Plots.figure(layout="double_column")
    axs = ICIW_Plots.make_square_subplots(
        fig, ax_width=5.5 * ICIW_Plots.cm2inch, ax_layout=(1, 2)
    )
    plot_true_data(axs, times_batch[0, :], y_true[0, :, :])
    plot_true_data(axs, times_batch[2, :], y_true[2, :, :])
    plot_true_data(axs, times_batch[4, :], y_true[4, :, :])
    plot_observed_data(axs, t_obs[2, :], y_obs[2, :, :])
    plt.show()

    # Inference using the single forward pass model and just one T per model
    model = bmodel(
        t_obs=t_obs[:, 2].squeeze(),
        T_obs=300,
        y_obs=y_obs[:, 2, :].squeeze(),
    )
    vars_list = list(model.values_to_rvs.keys())[:-1]
    with model:
        prior_idata = pm.sample_prior_predictive(samples=1000, random_seed=23489)
        posterior_idata = pm.sample(
            step=[pm.DEMetropolis(vars_list)],
            draws=3000,
            tune=1000,
            chains=6,
            # target_accept=0.9,
            random_seed=23489,
            return_inferencedata=True,
            # nuts_sampler="numpyro",
            cores=1,
        )

    fig = ICIW_Plots.figure(layout="double_column")
    axs = ICIW_Plots.make_square_subplots(
        fig, ax_width=5.5 * ICIW_Plots.cm2inch, ax_layout=(1, 2)
    )
    plot_observed_data(axs, t_obs[:, 2], y_obs[:, 2, :])
    plot_inference(
        axs,
        prior_idata,
        times=times,
        num_samples=25,
        lw=1,
        alpha=0.1,
    )
    plt.show()
    post_data = az.extract(
        posterior_idata,
        group="posterior",
        num_samples=25,
        var_names=["y_hat"],
    )
    # display(post_data)

    fig = ICIW_Plots.figure(layout="double_column")
    axs = ICIW_Plots.make_square_subplots(
        fig, ax_width=5.5 * ICIW_Plots.cm2inch, ax_layout=(1, 2)
    )

    for i in range(25):
        # print(f"Plotting sample {i+1}/25")
        plot_true_data(
            axs,
            t_obs[:, 2],
            post_data.values[:, :, i],
            lw=1,
            alpha=0.1,
        )
    plt.show()
