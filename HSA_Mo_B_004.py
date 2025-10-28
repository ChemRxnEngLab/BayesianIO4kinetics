"""
This file assume homoscedastic noise
maximal entropy on the priors
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
import pymc as pm
from scipy.constants import R
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import arviz as az
import xarray

SIDX = {
    "CO2": 0,
    "H2": 1,
    "H2O": 2,
    "CH4": 3,
    "I": 4,
}

TRUE_PARAMS = {
    "k0": 3.46e-4,  # mol/(bar·s·g_cat)
    "EA": 77.5e3,  # J/mol
    "A_OH": 0.50,  # bar^-0.5
    "H_OH": 22.4e3,  # J/mol
    "A_H2": 0.44,  # bar^-0.5
    "H_H2": -6.2e3,  # J/mol
    "A_mix": 0.88,  # bar^-0.5
    "H_mix": -10.0e3,  # J/mol
}


@jax.jit
def _kinetics_model(
    p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix
) -> jnp.ndarray:
    p_i = jnp.asarray(p_i)
    T = jnp.asarray(T)
    k0 = jnp.asarray(k0)
    Ea_k = jnp.asarray(Ea_k)
    A_oh = jnp.asarray(A_oh)
    dH_oh = jnp.asarray(dH_oh)
    A_h2 = jnp.asarray(A_h2)
    dH_h2 = jnp.asarray(dH_h2)
    A_mix = jnp.asarray(A_mix)
    dH_mix = jnp.asarray(dH_mix)
    k = k0 * jnp.exp(Ea_k / R * (1 / 555.0 - 1 / T))
    K_OH = A_oh * jnp.exp(dH_oh / R * (1 / 555.0 - 1 / T))
    K_H2 = A_h2 * jnp.exp(dH_h2 / R * (1 / 555.0 - 1 / T))
    K_mix = A_mix * jnp.exp(dH_mix / R * (1 / 555.0 - 1 / T))
    K_eq = 137 * T ** (-4) * jnp.exp(158700 / (R * T))

    driving_force = 1 - (p_i[:, SIDX["CH4"], None] * p_i[:, SIDX["H2O"], None] ** 2) / (
        p_i[:, SIDX["CO2"], None] * p_i[:, SIDX["H2"], None] ** 4 * K_eq
    )
    inhibition = (
        1
        + (K_OH * p_i[:, SIDX["H2O"], None]) / jnp.sqrt(p_i[:, SIDX["H2"], None])
        + K_H2 * jnp.sqrt(p_i[:, SIDX["H2"], None])
        + K_mix * jnp.sqrt(p_i[:, SIDX["CO2"], None])
    ) ** 2

    r = (
        k
        * jnp.sqrt(p_i[:, SIDX["H2"], None])
        * jnp.sqrt(p_i[:, SIDX["CO2"], None])
        * driving_force
        / inhibition
    )
    return r


@jax.jit
def _vjp_kinetics_model(p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix, gz):
    _, vjp_fun = jax.vjp(
        _kinetics_model,
        p_i,
        T,
        k0,
        Ea_k,
        A_oh,
        dH_oh,
        A_h2,
        dH_h2,
        A_mix,
        dH_mix,
    )
    return vjp_fun(gz)


class kinOp(Op):

    def make_node(
        self,
        p_i,
        T,
        k0,
        Ea_k,
        A_oh,
        dH_oh,
        A_h2,
        dH_h2,
        A_mix,
        dH_mix,
    ):
        p_i = pt.as_tensor_variable(p_i)
        T = pt.as_tensor_variable(T)
        k0 = pt.as_tensor_variable(k0)
        Ea_k = pt.as_tensor_variable(Ea_k)
        A_oh = pt.as_tensor_variable(A_oh)
        dH_oh = pt.as_tensor_variable(dH_oh)
        A_h2 = pt.as_tensor_variable(A_h2)
        dH_h2 = pt.as_tensor_variable(dH_h2)
        A_mix = pt.as_tensor_variable(A_mix)
        dH_mix = pt.as_tensor_variable(dH_mix)
        return Apply(
            self,
            [p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix],
            [pt.dmatrix()],
        )

    def perform(self, node, inputs, outputs):
        p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix = inputs
        # result = _forward_solve(y0, t, theta)
        result = _kinetics_model(
            p_i,
            T,
            k0,
            Ea_k,
            A_oh,
            dH_oh,
            A_h2,
            dH_h2,
            A_mix,
            dH_mix,
        )
        outputs[0][0] = np.asarray(result)

    def grad(self, inputs, output_grads):
        p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix = inputs
        (gz,) = output_grads
        (
            vjp_p_i,
            vjp_T,
            vjp_k0,
            vjp_Ea_k,
            vjp_A_oh,
            vjp_dH_oh,
            vjp_A_h2,
            vjp_dH_h2,
            vjp_A_mix,
            vjp_dH_mix,
        ) = VJPkinOp()(
            p_i,
            T,
            k0,
            Ea_k,
            A_oh,
            dH_oh,
            A_h2,
            dH_h2,
            A_mix,
            dH_mix,
            gz,
        )
        return [
            vjp_p_i,
            vjp_T,
            vjp_k0,
            vjp_Ea_k,
            vjp_A_oh,
            vjp_dH_oh,
            vjp_A_h2,
            vjp_dH_h2,
            vjp_A_mix,
            vjp_dH_mix,
        ]


class VJPkinOp(Op):
    def make_node(
        self,
        p_i,
        T,
        k0,
        Ea_k,
        A_oh,
        dH_oh,
        A_h2,
        dH_h2,
        A_mix,
        dH_mix,
        gz,
    ):
        p_i = pt.as_tensor_variable(p_i)
        T = pt.as_tensor_variable(T)
        k0 = pt.as_tensor_variable(k0)
        Ea_k = pt.as_tensor_variable(Ea_k)
        A_oh = pt.as_tensor_variable(A_oh)
        dH_oh = pt.as_tensor_variable(dH_oh)
        A_h2 = pt.as_tensor_variable(A_h2)
        dH_h2 = pt.as_tensor_variable(dH_h2)
        A_mix = pt.as_tensor_variable(A_mix)
        dH_mix = pt.as_tensor_variable(dH_mix)
        gz = pt.as_tensor_variable(gz)
        return Apply(
            self,
            [
                p_i,
                T,
                k0,
                Ea_k,
                A_oh,
                dH_oh,
                A_h2,
                dH_h2,
                A_mix,
                dH_mix,
                gz,
            ],
            [
                p_i.type(),  # gradient w.r.t. p_i (same shape as p_i)
                T.type(),  # gradient w.r.t. T (same shape as T)
                k0.type(),  # gradient w.r.t. k0 (same shape as k0, likely scalar)
                Ea_k.type(),  # gradient w.r.t. Ea_k (same shape as Ea_k, likely scalar)
                A_oh.type(),  # gradient w.r.t. A_oh (same shape as A_oh, likely scalar)
                dH_oh.type(),  # gradient w.r.t. dH_oh (same shape as dH_oh, likely scalar)
                A_h2.type(),  # gradient w.r.t. A_h2 (same shape as A_h2, likely scalar)
                dH_h2.type(),  # gradient w.r.t. dH_h2 (same shape as dH_h2, likely scalar)
                A_mix.type(),  # gradient w.r.t. A_mix (same shape as A_mix, likely scalar)
                dH_mix.type(),  # gradient w.r.t. dH_mix (same shape as dH_mix, likely scalar)
            ],
        )

    def perform(self, node, inputs, outputs):
        p_i, T, k0, Ea_k, A_oh, dH_oh, A_h2, dH_h2, A_mix, dH_mix, gz = inputs
        # vjp_y0, vjp_t, vjp_theta = _vjp_forward_solve(y0, t, theta, gz)
        (
            vjp_p_i,
            vjp_T,
            vjp_k0,
            vjp_Ea_k,
            vjp_A_oh,
            vjp_dH_oh,
            vjp_A_h2,
            vjp_dH_h2,
            vjp_A_mix,
            vjp_dH_mix,
        ) = _vjp_kinetics_model(
            p_i,
            T,
            k0,
            Ea_k,
            A_oh,
            dH_oh,
            A_h2,
            dH_h2,
            A_mix,
            dH_mix,
            gz,
        )

        outputs[0][0] = np.asarray(vjp_p_i)
        outputs[1][0] = np.asarray(vjp_T)
        outputs[2][0] = np.asarray(vjp_k0)
        outputs[3][0] = np.asarray(vjp_Ea_k)
        outputs[4][0] = np.asarray(vjp_A_oh)
        outputs[5][0] = np.asarray(vjp_dH_oh)
        outputs[6][0] = np.asarray(vjp_A_h2)
        outputs[7][0] = np.asarray(vjp_dH_h2)
        outputs[8][0] = np.asarray(vjp_A_mix)
        outputs[9][0] = np.asarray(vjp_dH_mix)


kinetics_model = kinOp()
vjp_kinetics_model = VJPkinOp()


@jax_funcify.register(kinOp)
def jax_funcify_kinetics_model(op, node, **kwargs):
    return _kinetics_model


@jax_funcify.register(VJPkinOp)
def jax_funcify_vjp_kinetics_model(op, node, **kwargs):
    return _vjp_kinetics_model


def input_p(p, c2h, f, c2i):
    x_i = jnp.zeros((p.shape[0], 5))
    x_i = x_i.at[:, SIDX["CO2"], None].set(1 / (1 + 1 / f + 1 / c2h + 1 / c2i))
    x_i = x_i.at[:, SIDX["H2"], None].set(x_i[:, SIDX["CO2"], None] / c2h)
    x_i = x_i.at[:, SIDX["CH4"], None].set(x_i[:, SIDX["CO2"], None] / f)
    x_i = x_i.at[:, SIDX["I"], None].set(x_i[:, SIDX["CO2"], None] / c2i)
    p_i = p * x_i
    return p_i


def bmodel(param_obs, y_obs):
    T, p, c2h, f, c2i = param_obs
    p_i = np.asarray(input_p(p, c2h, f, c2i))
    # T = pt.as_tensor(T)
    # p_i = pt.as_tensor(p_i)
    # y_obs = pt.as_tensor(y_obs)

    with pm.Model() as model:
        # Priors for unknown model parameters
        k0 = pm.Exponential("k0", scale=0.001)
        Ea_k = pm.Exponential("Ea_k", scale=60000)
        A_oh = pm.Exponential("A_oh", scale=1)
        dH_oh = pm.Exponential("dH_oh", scale=10000)
        A_h2 = pm.Exponential("A_h2", scale=1)
        dH_h2 = pm.Normal("dH_h2", mu=-10000, sigma=5000)
        # dH_h2 = -1 * pm.Exponential("dH_h2", scale=10000)
        A_mix = pm.Exponential("A_mix", scale=1)
        dH_mix = pm.Normal("dH_mix", mu=-10000, sigma=5000)
        # dH_mix = -1 * pm.Exponential("dH_mix", scale=10000)
        sigma_r = pm.HalfNormal("sigma_r", sigma=1e-5)
        y_hat = pm.Deterministic(
            "y_hat",
            kinetics_model(
                p_i,
                T,
                k0,
                Ea_k,
                A_oh,
                dH_oh,
                A_h2,
                dH_h2,
                A_mix,
                dH_mix,
            ),
        )

        LH2 = pm.Normal(
            "LH2",
            mu=y_hat,
            sigma=sigma_r,
            observed=y_obs,
        )

    return model


def main():
    test_params = {
        "p_i": input_p(jnp.array([4.0]), 1, 1000, 0.5).reshape(1, -1),
        "T": jnp.linspace(500.0, 600.0, 100).reshape(-1, 1),
        "k0": jnp.array([100, 200, 300]).reshape(1, -1),
        "Ea_k": jnp.array([TRUE_PARAMS["Ea_k"]]).reshape(1, -1),
        "A_oh": jnp.array([TRUE_PARAMS["A_oh"]]).reshape(1, -1),
        "dH_oh": jnp.array([TRUE_PARAMS["dH_oh"]]).reshape(1, -1),
        "A_h2": jnp.array([TRUE_PARAMS["A_h2"]]).reshape(1, -1),
        "dH_h2": jnp.array([TRUE_PARAMS["dH_h2"]]).reshape(1, -1),
        "A_mix": jnp.array([TRUE_PARAMS["A_mix"]]).reshape(1, -1),
        "dH_mix": jnp.array([TRUE_PARAMS["dH_mix"]]).reshape(1, -1),
    }

    test_res = _kinetics_model(**test_params)
    print(test_res.shape)


if __name__ == "__main__":
    main()
