import os

# Configure PyTensor to use JAX backend and avoid C compilation
# os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_compile,mode=JAX"
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import jax
import pytensor

# Set PyTensor to use JAX mode (redundant with env var but explicit)
pytensor.config.mode = "JAX"
# pytensor.config.optimizer = "fast_compile"

import numpy as np
import matplotlib.pyplot as plt

import pymc as pm

import arviz as az
from scipy.optimize import minimize

from HSA_Mo_B_004 import (
    SIDX,
    TRUE_PARAMS,
    _kinetics_model,
    input_p,
    bmodel,
)


# %%
def post_model_over_T(proposal, idata, key="posterior"):

    test_params = {
        "p_i": input_p(
            jnp.array(proposal[1]).reshape(1, 1),
            jnp.array(proposal[2]).reshape(1, 1),
            jnp.array(proposal[3]).reshape(1, 1),
            jnp.array(proposal[4]).reshape(1, 1),
        ).reshape(1, -1),
        "T": jnp.linspace(500.0, 600.0, 100).reshape(-1, 1),
        "k0": idata[key]["k0"].values.reshape(1, -1),
        "Ea_k": idata[key]["Ea_k"].values.reshape(1, -1),
        "A_oh": idata[key]["A_oh"].values.reshape(1, -1),
        "dH_oh": idata[key]["dH_oh"].values.reshape(1, -1),
        "A_h2": idata[key]["A_h2"].values.reshape(1, -1),
        "dH_h2": idata[key]["dH_h2"].values.reshape(1, -1),
        "A_mix": idata[key]["A_mix"].values.reshape(1, -1),
        "dH_mix": idata[key]["dH_mix"].values.reshape(1, -1),
    }

    test_res = _kinetics_model(**test_params)
    return test_res


def true_model_over_T(proposal):
    true_res_T = _kinetics_model(
        input_p(
            jnp.array(proposal[1]).reshape(1, 1),
            jnp.array(proposal[2]).reshape(1, 1),
            jnp.array(proposal[3]).reshape(1, 1),
            jnp.array(proposal[4]).reshape(1, 1),
        ).reshape(1, -1),
        jnp.linspace(500.0, 600.0, 100).reshape(-1, 1),
        *TRUE_PARAMS.values(),
    )
    return true_res_T


def plot_post_model_at_proposal(proposal, test_res, true_res, Ts):
    plt.plot(
        Ts,
        test_res.mean(axis=1),
    )
    plt.plot(
        Ts,
        true_res.squeeze(),
        color="black",
        linestyle="--",
    )
    plt.fill_between(
        x=Ts,
        y1=np.percentile(test_res, 2.5, axis=1),
        y2=np.percentile(test_res, 97.5, axis=1),
        alpha=0.3,
        zorder=0,
    )
    ax = plt.gca()
    ax.text(
        0.05,
        0.95,
        f"pred. @ p={proposal[1]:.2f}, F= {proposal[2]:.2f}, C= {proposal[3]:.2f}, H2= {proposal[4]:.2f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax2 = ax.twinx()
    ax2.plot(
        Ts,
        np.percentile(test_res, 97.5, axis=1) - np.percentile(test_res, 2.5, axis=1),
        color="orange",
    )


def observe_at_proposal(proposals):
    T = jnp.array([proposals[:, 0]], dtype=jnp.float32).reshape(-1, 1)
    p = jnp.array([proposals[:, 1]], dtype=jnp.float32).reshape(-1, 1)
    c2h = jnp.array([proposals[:, 2]], dtype=jnp.float32).reshape(-1, 1)
    f = jnp.array([proposals[:, 3]], dtype=jnp.float32).reshape(-1, 1)
    c2i = jnp.array([proposals[:, 4]], dtype=jnp.float32).reshape(-1, 1)

    p_i = input_p(p, c2h, f, c2i)

    r = _kinetics_model(p_i, T, *TRUE_PARAMS.values())
    r_obs = r + jax.random.normal(jax.random.PRNGKey(1), r.shape) * 1e-5
    return T, p, c2h, f, c2i, r_obs


def max_func(x, idata, key="posterior"):
    T = x[0].reshape(1, 1)
    p = x[1].reshape(1, 1)
    c2h = x[2].reshape(1, 1)
    f = x[3].reshape(1, 1)
    c2i = x[4].reshape(1, 1)
    p_i = input_p(p, c2h, f, c2i)
    res = _kinetics_model(
        p_i,
        T,
        idata[key]["k0"].values.reshape(1, -1),
        idata[key]["Ea_k"].values.reshape(1, -1),
        idata[key]["A_oh"].values.reshape(1, -1),
        idata[key]["dH_oh"].values.reshape(1, -1),
        idata[key]["A_h2"].values.reshape(1, -1),
        idata[key]["dH_h2"].values.reshape(1, -1),
        idata[key]["A_mix"].values.reshape(1, -1),
        idata[key]["dH_mix"].values.reshape(1, -1),
    )
    HDI = jnp.percentile(res, 97.5) - jnp.percentile(res, 2.5)
    return -HDI


if __name__ == "__main__":
    T = jnp.array([560, 570, 580], dtype=jnp.float32).reshape(-1, 1)
    p = jnp.array([4, 5, 6], dtype=jnp.float32).reshape(-1, 1)
    c2h = jnp.array([1], dtype=jnp.float32).reshape(-1, 1)
    f = jnp.array([1000], dtype=jnp.float32).reshape(-1, 1)
    c2i = jnp.array([0.5], dtype=jnp.float32).reshape(-1, 1)

    p_i = input_p(p, c2h, f, c2i)
    print(p_i)

    r = _kinetics_model(p_i, T, *TRUE_PARAMS.values())
    print(r)
    r_obs = r + jax.random.normal(jax.random.PRNGKey(1), r.shape) * 1e-5
    print(r_obs)

    model = bmodel(
        (
            np.asarray(T),
            np.asarray(p),
            np.asarray(c2h),
            np.asarray(f),
            np.asarray(c2i),
        ),
        np.asarray(r_obs),
    )

    with model:
        prior_idata = pm.sample_prior_predictive(samples=1000)

    proposals = np.array(
        [
            [500.0, 4.0, 0.5, 1000.0, 0.5],
            [600.0, 4.0, 0.5, 1000.0, 0.5],
            [500.0, 20.0, 0.5, 1000.0, 0.5],
            [600.0, 20.0, 0.5, 1000.0, 0.5],
        ]
    )

    # ## Automatic Iteration

    n_iterations = 8
    i_datas = []
    max_HDIs = []

    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")
        T, p, c2h, f, c2i, r_obs = observe_at_proposal(proposals)
        model = bmodel(
            (
                np.asarray(T),
                np.asarray(p),
                np.asarray(c2h),
                np.asarray(f),
                np.asarray(c2i),
            ),
            np.asarray(r_obs),
        )
        with model:
            idata = pm.sample(
                tune=1000,
                draws=4000,
                chains=4,
                cores=1,  # Use single core to avoid multiprocessing issues
                target_accept=0.9,
                return_inferencedata=True,
            )
            i_datas.append(idata)
        sol = minimize(
            max_func,
            jnp.array([555.0, 15.0, 1.0, 1000.0, 0.5]),
            args=(idata, "posterior"),
            bounds=((500, 600), (2, 20), (0.5, 2), (1, 1000), (0.1, 1)),
            method="Nelder-Mead",
        )
        max_HDIs.append(-sol.fun)
        proposals = np.append(proposals, np.atleast_2d(sol.x), axis=0)
        print("Next Proposal:\n", sol.x)
