from pyexpat import model
from scipy.integrate import odeint
import pytensor as pt
from pytensor.compile.ops import as_op
import numpy as np
import matplotlib.pyplot as plt
import ICIW_Plots
import pymc as pm
import arviz as az
from scipy.constants import R
import xarray
from numba import njit
from HSA_Cam_A_005_lib import Chem

pt.config.mode = "NUMBA"

plt.style.use("ICIWstyle")

# For reproducibility
np.random.seed(23489)

y0_true = (0.0, 0.0, 10.0, 0.0)
theta_true = (10e5, 300, 3e-3, 0.0001, 40000)
# sigma = 0
sigma = 0.5


def bmodel(t_obs, y_obs):
    with pm.Model() as model:
        kla = pm.Lognormal("kla", mu=np.log(3e-3), sigma=1)
        A0 = pm.Lognormal("A0", mu=np.log(1e-4), sigma=1)
        EA = pm.Normal("EA", mu=60000, sigma=5000)
        sigma = pm.HalfNormal("sigma", sigma=1)

        y_hat = pm.Deterministic(
            "y_hat",
            Chem.pt_forward(
                pm.math.stack([*y0_true]),
                pm.math.stack([*t_obs]),
                pm.math.stack((theta_true[0], theta_true[1], kla, A0, EA)),
            ),
        )
        # just observe the H2 consumption

        LH2 = pm.Normal("LH2", mu=y_hat[:, 0] * 1000, sigma=sigma, observed=y_obs[:, 0])

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


def plot_inference(
    ax,
    trace,
    times,
    group="prior",
    theta_indep=None,
    num_samples=25,
    **kwargs,
):
    trace_df: xarray.Dataset = az.extract(trace, group=group, num_samples=num_samples)

    for row_idx in range(num_samples):
        print(f"Plotting sample {row_idx+1}/{num_samples}")
        plot_model_trace(
            ax, trace_df, times, row_idx, theta_indep=theta_indep, **kwargs
        )


def plot_model_trace(ax, trace_df, times, row_idx, theta_indep=None, lw=1, alpha=0.2):
    row = trace_df.isel(sample=row_idx)[["kla", "A0", "EA"]].to_array().values
    if theta_indep is None:
        theta_indep = (theta_true[0], theta_true[1])
    theta = (*theta_indep, *row)
    print(f"theta: {theta}")
    y_prior = odeint(func=Chem.reaction, y0=y0_true, t=times, args=(theta,))
    y_prior[:, 0] *= 1000  # convert m3 to L H2 used (up to a constant atleast)
    plot_true_data(ax, times, y_prior, lw=lw, alpha=alpha)


if __name__ == "__main__":
    # Times for observation
    times = np.linspace(0, 1 * 60 * 60, 100)

    y_true = xarray.DataArray(
        np.zeros((3, len(times), 4)),
        dims=["temperature", "time", "state"],
        coords={
            "temperature": [280, 300, 320],
            "time": times,
            "state": ["n_g_con", "c_Al", "c_Bl", "c_Cl"],
        },
    )

    for i, T in enumerate([280, 300, 320]):
        theta = (
            theta_true[0],
            T,
            theta_true[2],
            theta_true[3],
            theta_true[4],
        )
        y_true[i, :, :] = odeint(Chem.reaction, t=times, y0=y0_true, args=(theta,))
    y_true[:, :, 0] *= 1000  # convert m3 to L H2 used (up to a constant atleast)

    # Add noise
    t_obs = times[::10]
    y_obs = np.random.normal(y_true[1, :, :], sigma)[
        ::10, :
    ]  # downsample and add noise

    ## Plotting the base model
    fig = ICIW_Plots.figure(layout="double_column")
    axs = ICIW_Plots.make_square_subplots(
        fig, ax_width=5.5 * ICIW_Plots.cm2inch, ax_layout=(1, 2)
    )
    plot_true_data(axs, times, y_true[1, :, :])
    plot_true_data(axs, times, y_true[2, :, :])
    plot_true_data(axs, times, y_true[0, :, :])
    plot_observed_data(axs, t_obs, y_obs)
    plt.show()

    model = bmodel(t_obs=t_obs, y_obs=y_obs)
    vars_list = list(model.values_to_rvs.keys())[:-1]
    with model:
        prior_idata = pm.sample_prior_predictive(samples=1000, random_seed=23489)
        posterior_idata = pm.sample(
            step=[pm.DEMetropolis(vars_list)],
            draws=3000,
            tune=1000,
            chains=8,
            # target_accept=0.9,
            random_seed=23489,
            return_inferencedata=True,
        )

    fig = ICIW_Plots.figure(layout="double_column")
    axs = ICIW_Plots.make_square_subplots(
        fig, ax_width=5.5 * ICIW_Plots.cm2inch, ax_layout=(1, 2)
    )
    plot_observed_data(axs, t_obs, y_obs)
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
            t_obs,
            post_data.values[:, :, i],
            lw=1,
            alpha=0.1,
        )
    plt.show()
