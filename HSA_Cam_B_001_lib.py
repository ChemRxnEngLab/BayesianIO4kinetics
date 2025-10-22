import jax
from jax import config, jit
import jax.numpy as jnp
import diffrax as dfx
import pytensor as pt
import numpy as np
from scipy.constants import R
from pytensor.graph import Apply, Op
from pytensor.gradient import verify_grad
from pytensor.link.jax.dispatch import jax_funcify
import time
from ReactionsPy.ReacNets import Jax_Reaction_Network
from ReactionsPy.JaxTempDeps import (
    Arrhenius_reparameterized as Jax_Arrhenius_reparameterized,
)
import numpy as np

config.update("jax_enable_x64", True)

mu = np.array(
    [
        [+0, +0, -1, -1, +0, +2, +0, +0, +0, +0, +0, +0, +0, +0],
        [+0, +0, +1, +1, +0, -2, +0, +0, +0, +0, +0, +0, +0, +0],
        [+0, +0, +0, +1, +0, -2, -1, +1, +0, +0, +0, +0, +0, +0],
        [+0, +0, +0, -1, +0, +2, +1, -1, +0, +0, +0, +0, +0, +0],
        [+0, +0, +0, +1, +0, -2, +0, -1, +1, +0, +0, +0, +0, +0],
        [+0, +0, +0, -1, +0, +2, +0, +1, -1, +0, +0, +0, +0, +0],
        [+0, +0, +0, +1, +0, -2, +0, +0, -1, +1, +0, +0, +0, +0],
        [+0, +0, +0, -1, +0, +2, +0, +0, +1, -1, +0, +0, +0, +0],
        [+0, +0, +0, +1, +0, -2, +0, +0, +0, -1, +1, +0, +0, +0],
        [+0, +0, +0, -1, +0, +2, +0, +0, +0, +1, -1, +0, +0, +0],
        [+0, +0, +0, +0, -1, +0, +0, +0, +0, +0, -1, +1, +0, +0],
        [+0, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, -1, +0, +0],
        [+0, +0, +0, +0, +1, +0, +0, -1, +0, +0, +0, -1, +1, +0],
        [+0, +0, +0, +0, -1, +0, +0, +1, +0, +0, +0, +1, -1, +0],
        [+0, +0, +0, +0, +1, +0, +0, +0, +0, -1, +0, -1, +0, +1],
        [+0, +0, +0, +0, -1, +0, +0, +0, +0, +1, +0, +1, +0, -1],
    ]
)

nu = np.where(mu < 0, -mu, 0)
nu[[2, 4, 6, 8], 5] = 1

jax_rxns = Jax_Reaction_Network(
    species=[
        "H2O",
        "L",
        "H2",
        "(s)2",
        "(X)",
        "H(s)",
        "(CHC)R(CCH)",
        "(CHC)R(CHCH2)",
        "(CHC)R(CH2CH3)",
        "(CH2CH)R(CH2CH3)",
        "(CH3CH2)R(CH2CH3)",
        "(CH3CH2)R(CH2CH3)(X)",
        "(CH3CH2)R(C4H8)R(CCH)",
        "(CH3CH2)R(C4H8)R(CH2CH3)",
    ],
    stoch_coeff_mat=mu,
    reaction_orders=nu,
    temp_dep=Jax_Arrhenius_reparameterized(
        k_0=np.array(
            [
                130,  # 0.00013,
                10,  # 1e-5,
                3.725,  # 0.67,  # EA
                0,
                136.83,  # 5e-5,
                0,
                3.0,  # 0.67,  # EA
                0,
                50,  # 5e-5,
                0,
                5.0122,  # 1e-6,
                0.00473,  # 0.001,
                1.0,  # 5.95 # EA # half because of two dimers formed
                0,
                1.0,  # 5.95,  # EA # half because of two dimers formed
                0,
            ]
        ),
        E_A=np.array(
            [
                0,
                0,
                57822,  # 30000,
                0,
                0,
                0,
                30000,
                0,
                0,
                0,
                0,
                0,
                38000,
                0,
                38000,
                0,
            ]
        ),
        T_ref=np.full((16,), 293),
    ),
)

sym_dict = {
    6: "RDY",
    7: "RYE",
    8: "RYA",
    9: "REA",
    10: "RDEt",
    11: "RDEt(X)",
    12: "dimerY",
    13: "dimerA",
}

for i, s in enumerate(jax_rxns.species):
    if i in sym_dict:
        s.sym = sym_dict[i]

N_SPECIES = 14

SIDX = {
    "H2O": 0,
    "solvent": 1,
    "H2": 2,
    "(s)2": 3,
    "(X)": 4,
    "H(s)": 5,
    "RDY": 6,
    "RYE": 7,
    "RYA": 8,
    "REA": 9,
    "RDEt": 10,
    "RDEt(X)": 11,
    "dimerY": 12,
    "dimerA": 13,
    "RDY[S]": 14,
    "conH2[G]": 15,
    "T": 16,
    "a": 17,
}
N_STATES = len(SIDX)
PIDX = {
    "p": 0,
    "RDY[S]0": 1,
    "k_deact": 2,
    "EA_deact": 3,
    "n_deact": 4,
}
N_PARAMS = len(PIDX)

# density liquid
rho_l = 1100.0  # kg/L
# solid density
rho_s = 1000.0  # kg/m^3????

# initial masses of all species in the liquid phase
# m_li0 = np.zeros((N_SPECIES), dtype=np.float64)
# m_li0[SIDX["H2O"]] = param_list[88]  # H20
# m_li0[SIDX["solvent"]] = param_list[89]  # solvent
# m_li0[SIDX["(s)2"]] = param_list[90]  # Cat2
# m_li0[SIDX["(X)"]] = param_list[91]  # CatX

# m_cat = param_list[90]  # initial mass of catalyst
# m_s0 = param_list[92]  # initial mass of solid RDY

# m_lt0 = np.sum(m_li0)
# V_lt = m_lt0 / rho_l  # L Volume is constant

# mass transfer coefficients
## gas-liquid
k_La = 0.25  # 10 1/s
## solid-liquid
k_Sa_0 = 10.0  # corrected for the units

## Henry's law gas-liquid
# k_H = param_list[17]
k_H = 7.7e-6 * 1e-3  # mol/(m3 Pa) -> mol/(L Pa)

# molar masses
M = jnp.array(
    [
        18.0,
        60.0,
        2.0,
        100.0,
        100.0,
        51.0,
        150.0,
        152.0,
        154.0,
        156.0,
        158.0,
        258.0,
        310.0,
        314.0,
    ],
    dtype=jnp.float64,
)


def dissolution_equilibrium(m_li):
    w_RDY_l = m2w(m_li)[SIDX["RDEt"]]
    return 0.00159 * jnp.exp(-0.1126 * (100.0 * w_RDY_l) ** 0.83)


def m2c(m_li):
    c_li = m_li / (M * m2V(m_li))
    return c_li


def m2V(m_li):
    V_lt = jnp.sum(m_li) / rho_l
    return V_lt


def m2w(m_li):
    w_li = jnp.clip(m_li, min=0) / jnp.sum(m_li)
    return w_li


def _ODE_func(t, y, params):
    m_li = y[0:N_SPECIES]
    m_s = y[SIDX["RDY[S]"]]
    con_H2 = y[SIDX["conH2[G]"]]
    T = y[SIDX["T"]]
    a = y[SIDX["a"]]
    p = params[PIDX["p"]]
    m_s0 = params[PIDX["RDY[S]0"]]
    k_deact = params[PIDX["k_deact"]]
    EA_deact = params[PIDX["EA_deact"]]
    n_deact = params[PIDX["n_deact"]]

    V_lt = m2V(m_li)
    c_li = m2c(m_li)
    c_li = jnp.clip(c_li, min=1e-12)

    # deactivation mechanism
    r_deact = jnp.where(
        a <= 1e-3,
        0.0,
        k_deact * jnp.exp((-EA_deact / R) * (1 / T - 1 / 293)) * a**n_deact,
    )

    a = jnp.clip(a, a_min=0)

    s_i = a * jax_rxns.formation_rates(c_li, T).squeeze()

    s_i = jnp.where(
        jnp.logical_and(c_li <= 1e-12, s_i < 0),
        0,
        s_i,
    )

    s_gli = jnp.zeros_like(s_i)
    s_sli = jnp.zeros_like(s_i)

    # gas-liquid mass transfer
    c_H2_li_eq = k_H * p
    s_gli = s_gli.at[SIDX["H2"]].set(k_La * (c_H2_li_eq - c_li[SIDX["H2"]]))

    # solid-liquid mass transfer
    c_RDY_eq = dissolution_equilibrium(m_li)
    # return zero mass transfer if the solid is depleted
    k_Sa = jnp.where(
        m_s <= 1e-5,
        0.0,
        k_Sa_0 * (m_s / m_s0) ** (2 / 3),
    )
    s_sli = s_sli.at[SIDX["RDY"]].set(k_Sa * (c_RDY_eq - c_li[SIDX["RDY"]]))

    dm_li = (s_i + s_gli + s_sli) * M * V_lt
    dm_s = jnp.atleast_1d(-s_sli[SIDX["RDY"]] * M[SIDX["RDY"]] * V_lt)
    dcon_H2 = jnp.atleast_1d(s_gli[SIDX["H2"]] * M[SIDX["H2"]] * V_lt)
    dT = jnp.zeros((1,), dtype=jnp.float64)
    da = jnp.atleast_1d(-r_deact)

    return jnp.concatenate([dm_li, dm_s, dcon_H2, dT, da])


ODE_func = jit(_ODE_func)

vec_ODE_func = jax.vmap(ODE_func, in_axes=(None, 0, 0))


def _forward_solve(y0, t, theta):
    # y0_jax = jnp.asarray(y0)
    # t_jax = jnp.asarray(t)
    # theta_jax = jnp.asarray(theta)

    # Define ODE term for Diffrax
    ode_term = dfx.ODETerm(ODE_func)

    # Set up solver - Tsit5 is a high-order Runge-Kutta method
    solver = dfx.Tsit5()
    # solver = dfx.Kvaerno4()

    # Define step size controller for adaptive stepping
    # stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)
    stepsize_controller = dfx.ConstantStepSize()

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        ode_term,
        solver,
        t0=t[0],
        t1=t[-1],
        # dt0=None,  # Let Diffrax choose initial step size
        dt0=1.0e-1,
        y0=y0,
        args=theta,
        saveat=dfx.SaveAt(ts=t),
        stepsize_controller=stepsize_controller,
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=50),
        max_steps=None,
    )

    return solution.ys


_jitted_forward_solve = jit(_forward_solve)


def _vjp_forward_solve(y0, t, theta, gz):
    """Vector-Jacobian product for forward solve."""
    _, vjp_fn = jax.vjp(_forward_solve, y0, t, theta)
    return vjp_fn(gz)


_jitted_vjp_forward_solve = jit(_vjp_forward_solve)

# =====================================================
# TRULY VECTORIZED BATCH IMPLEMENTATION
# =====================================================


@jit
def _vec_ODE_func_batch(t, y_flat, params_batch):
    """
    Truly vectorized reaction function using vmap on the base reaction function.
    This is cleaner and leverages JAX's optimized vmap implementation.

    Parameters:
    -----------
    t : float
        Time point
    y_flat : array, shape (batch_size * n_states,)
        Flattened state vector for all batch elements
    params_batch : array, shape (batch_size, n_params)
        Parameter array for each batch element

    Returns:
    --------
    dydt_flat : array, shape (batch_size * n_states,)
        Flattened time derivatives for all batch elements
    """
    batch_size = params_batch.shape[0]

    # Reshape flat state vector back to batch format
    y_batch = y_flat.reshape(batch_size, N_STATES)

    # Use vmap to vectorize the base reaction function
    dydt_batch = vec_ODE_func(t, y_batch, params_batch)

    return dydt_batch.flatten()


def _vec_forward_solve(y0_batch, t_eval, params_batch):
    """
    Truly vectorized ODE solver that treats the batch as a single large ODE system.
    This achieves better performance by allowing full vectorization within the solver.

    Parameters:
    -----------
    y0_batch : array, shape (batch_size, n_states)
        Initial conditions for each batch element
    t_eval : array, shape (n_times,)
        Time points for evaluation (same for all batch elements)
    params_batch : array, shape (batch_size, n_params)
        Parameters for each batch element

    Returns:
    --------
    solutions : array, shape (batch_size, n_times, n_states)
        Solutions for all batch elements
    """
    # Convert inputs to JAX arrays
    y0_batch_jax = jnp.asarray(y0_batch)
    t_eval_jax = jnp.asarray(t_eval)
    params_batch_jax = jnp.asarray(params_batch)

    batch_size, _ = y0_batch_jax.shape

    # Flatten initial conditions for the vectorized solver
    y0_flat = y0_batch_jax.flatten()

    # Define ODE term for Diffrax with vectorized reaction
    ode_term = dfx.ODETerm(_vec_ODE_func_batch)

    # Set up solver - Tsit5 is a high-order Runge-Kutta method
    solver = dfx.Tsit5()

    # Define step size controller for adaptive stepping
    stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)

    # Solve the vectorized ODE system
    solution = dfx.diffeqsolve(
        ode_term,
        solver,
        t0=t_eval_jax[0],
        t1=t_eval_jax[-1],
        dt0=None,  # Let Diffrax choose initial step size
        y0=y0_flat,
        args=params_batch_jax,
        saveat=dfx.SaveAt(ts=t_eval_jax),
        stepsize_controller=stepsize_controller,
        max_steps=2**16,
    )

    # Reshape solution back to batch format
    n_times = len(t_eval_jax)
    solutions_flat = solution.ys  # shape (n_times, batch_size * n_states)
    solutions_batch = solutions_flat.reshape(n_times, batch_size, N_STATES)

    # Transpose to get (batch_size, n_times, n_states)
    return jnp.transpose(solutions_batch, (1, 0, 2))


_jitted_vec_forward_solve = jit(_vec_forward_solve)


def _vjp_vec_forward_solve(y0_batch, t_eval, params_batch, gz_batch):
    """Vector-Jacobian product for vectorized forward solve."""
    _, vjp_fn = jax.vjp(_vec_forward_solve, y0_batch, t_eval, params_batch)
    return vjp_fn(gz_batch)


_jitted_vjp_vec_forward_solve = jit(_vjp_vec_forward_solve)


class SolOp(Op):
    def make_node(self, y0, t, theta):
        y0 = pt.tensor.as_tensor_variable(y0)
        t = pt.tensor.as_tensor_variable(t)
        theta = pt.tensor.as_tensor_variable(theta)
        return Apply(self, [y0, t, theta], [pt.tensor.dmatrix()])

    def perform(self, node, inputs, outputs):
        y0, t, theta = inputs
        # result = _forward_solve(y0, t, theta)
        result = _jitted_forward_solve(y0, t, theta)
        outputs[0][0] = np.asarray(result)

    def grad(self, inputs, output_grads):
        y0, t, theta = inputs
        (gz,) = output_grads
        vjp_y0, vjp_t, vjp_theta = VJPSolOp()(y0, t, theta, gz)
        return [vjp_y0, vjp_t, vjp_theta]


class VJPSolOp(Op):
    def make_node(self, y0, t, theta, gz):
        y0 = pt.tensor.as_tensor_variable(y0)
        t = pt.tensor.as_tensor_variable(t)
        theta = pt.tensor.as_tensor_variable(theta)
        gz = pt.tensor.as_tensor_variable(gz)
        return Apply(
            self,
            [y0, t, theta, gz],
            [pt.tensor.dvector(), pt.tensor.dvector(), pt.tensor.dvector()],
        )

    def perform(self, node, inputs, outputs):
        y0, t, theta, gz = inputs
        # vjp_y0, vjp_t, vjp_theta = _vjp_forward_solve(y0, t, theta, gz)
        vjp_y0, vjp_t, vjp_theta = _jitted_vjp_forward_solve(y0, t, theta, gz)
        outputs[0][0] = np.asarray(vjp_y0)
        outputs[1][0] = np.asarray(vjp_t)
        outputs[2][0] = np.asarray(vjp_theta)


class VecSolOp(Op):
    """
    PyTensor Op for truly vectorized batch ODE solving.
    More efficient than ForwardSolveBatchOp as it treats the batch as a single ODE system.
    """

    def make_node(self, y0_batch, t, params_batch):
        y0_batch = pt.tensor.as_tensor_variable(y0_batch)  # (batch_size, n_states)
        t = pt.tensor.as_tensor_variable(t)  # (n_times,)
        params_batch = pt.tensor.as_tensor_variable(
            params_batch
        )  # (batch_size, n_params)
        outputs = [pt.tensor.dtensor3()]  # (batch_size, n_times, n_states)
        return Apply(self, [y0_batch, t, params_batch], outputs)

    def perform(self, node, inputs, outputs):
        y0_batch, t, params_batch = inputs
        # result = _vec_forward_solve(y0_batch, t, params_batch)
        result = _jitted_vec_forward_solve(y0_batch, t, params_batch)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs, output_grads):
        y0_batch, t, params_batch = inputs
        (gz,) = output_grads
        vjp_y0, vjp_t, vjp_params = VJPVecSolOp()(y0_batch, t, params_batch, gz)
        return [vjp_y0, vjp_t, vjp_params]


class VJPVecSolOp(Op):
    """
    Vector-Jacobian product Op for the vectorized batch solver.
    """

    def make_node(self, y0_batch, t, params_batch, gz):
        y0_batch = pt.tensor.as_tensor_variable(y0_batch)
        t = pt.tensor.as_tensor_variable(t)
        params_batch = pt.tensor.as_tensor_variable(params_batch)
        gz = pt.tensor.as_tensor_variable(gz)
        return Apply(
            self,
            [y0_batch, t, params_batch, gz],
            [pt.tensor.dmatrix(), pt.tensor.dvector(), pt.tensor.dmatrix()],
        )

    def perform(self, node, inputs, outputs):
        y0_batch, t, params_batch, gz = inputs
        # vjp_y0, vjp_t, vjp_params = _vjp_vec_forward_solve(
        #     y0_batch, t, params_batch, gz
        # )
        vjp_y0, vjp_t, vjp_params = _jitted_vjp_vec_forward_solve(
            y0_batch, t, params_batch, gz
        )
        outputs[0][0] = np.asarray(vjp_y0)
        outputs[1][0] = np.asarray(vjp_t)
        outputs[2][0] = np.asarray(vjp_params)


forward_solve = SolOp()
vec_forward_solve = VecSolOp()
vjp_forward_solve = VJPSolOp()
vjp_vec_forward_solve = VJPVecSolOp()


@jax_funcify.register(SolOp)
def jax_funcify_forward_solve(op, node, **kwargs):
    return _forward_solve


@jax_funcify.register(VJPSolOp)
def jax_funcify_vjp_forward_solve(op, node, **kwargs):
    return _vjp_forward_solve


@jax_funcify.register(VecSolOp)
def jax_funcify_forward_solve_vectorized(op, node, **kwargs):
    return _vec_forward_solve


@jax_funcify.register(VJPVecSolOp)
def jax_funcify_vjp_forward_solve_vectorized(op, node, **kwargs):
    return _vjp_vec_forward_solve
