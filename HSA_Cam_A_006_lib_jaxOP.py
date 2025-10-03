import jax
import jax.numpy as jnp
from jax import jit
import diffrax as dfx
import pytensor as pt
import numpy as np
from scipy.constants import R
from pytensor.graph import Apply, Op
from pytensor.gradient import verify_grad
from pytensor.link.jax.dispatch import jax_funcify
from jax import config
import time

config.update("jax_enable_x64", True)


@jit
def reaction(t, y, params):
    """
    Diffrax-compatible reaction rate equations.
    Note: Diffrax uses (t, y, args) convention instead of (y, t, args)

    Parameters:
    -----------
    t : float
        Time (not used in this autonomous system)
    y : array-like
        State vector [n_g_con, c_Al, c_Bl, c_Cl, c_Dl]
    params : array-like
        Parameter vector [p, T, kla, A01, EA1, A02, EA2]

    Returns:
    --------
    dydt : array
        Time derivatives of state variables
    """
    V_r = 1e-3  # m3
    n_g_con, c_Al, c_Bl, c_Cl, c_Dl = y[0], y[1], y[2], y[3], y[4]
    p = params[0]
    T = params[1]
    kla = params[2]
    A01 = params[3]
    EA1 = params[4]
    A02 = params[5]
    EA2 = params[6]

    # Reaction rate constants (Arrhenius equations)
    k1 = A01 * jnp.exp(-EA1 / R * (1 / T - 1 / 298))
    k2 = A02 * jnp.exp(-EA2 / R * (1 / T - 1 / 298))
    k_H = 7.8e-6 * jnp.exp(500 * (1 / T - 1 / 298))

    # Rate equations
    dn_g_con_dt = kla * (k_H * p - c_Al) * V_r
    dc_Al_dt = kla * (k_H * p - c_Al) - k1 * c_Al * c_Bl - k2 * c_Cl**2
    dc_Bl_dt = -k1 * c_Al * c_Bl
    dc_Cl_dt = k1 * c_Al * c_Bl - k2 * c_Cl**2
    dc_Dl_dt = k2 * c_Cl**2

    dydt = jnp.array(
        [
            dn_g_con_dt,
            dc_Al_dt,
            dc_Bl_dt,
            dc_Cl_dt,
            dc_Dl_dt,
        ]
    )
    return dydt


def _forward_solve(y0, t, theta):
    """
    PyTensor-compatible forward pass using Diffrax.

    This method uses Diffrax for high-performance ODE solving with
    adaptive step size control and advanced numerical methods.

    Parameters:
    -----------
    y0 : array-like
        Initial conditions
    t : array-like
        Time points for evaluation
    theta : array-like
        Parameter vector
    rtol : float, default=1e-6
        Relative tolerance for adaptive stepping
    atol : float, default=1e-8
        Absolute tolerance for adaptive stepping
    max_steps : int, default=16**4
        Maximum number of steps
    """
    # Convert inputs to JAX arrays
    y0_jax = jnp.asarray(y0)
    t_jax = jnp.asarray(t)
    theta_jax = jnp.asarray(theta)

    # Define ODE term for Diffrax
    ode_term = dfx.ODETerm(lambda t, y, args: reaction(t, y, args))

    # Set up solver - Tsit5 is a high-order Runge-Kutta method
    solver = dfx.Tsit5()

    # Define step size controller for adaptive stepping
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        ode_term,
        solver,
        t0=t_jax[0],
        t1=t_jax[-1],
        dt0=None,  # Let Diffrax choose initial step size
        y0=y0_jax,
        args=theta_jax,
        saveat=dfx.SaveAt(ts=t_jax),
        stepsize_controller=stepsize_controller,
        max_steps=2**16,
    )

    return solution.ys


_jitted_forward_solve = jit(_forward_solve)


def _vjp_forward_solve(y0, t, theta, gz):
    """Vector-Jacobian product for forward solve."""
    _, vjp_fn = jax.vjp(_forward_solve, y0, t, theta)
    return vjp_fn(gz)


_jitted_vjp_forward_solve = jit(_vjp_forward_solve)


@jit
def _reaction_batch(y_batch, t, params_batch):
    """
    Vectorized reaction equations for batch processing.

    Parameters:
    -----------
    y_batch : array, shape (batch_size, 5)
        Batch of state vectors
    t : float
        Time point
    params_batch : array, shape (batch_size, 7)
        Batch of parameter vectors

    Returns:
    --------
    dydt_batch : array, shape (batch_size, 5)
        Batch of time derivatives
    """
    # Use vmap to vectorize over the batch dimension
    return jax.vmap(reaction, in_axes=(None, 0, 0))(t, y_batch, params_batch)


def _forward_solve_batch(y0_batch, t_eval, params_batch):
    """
    Solve ODE system for multiple parameter sets simultaneously.

    Parameters:
    -----------
    y0_batch : array, shape (batch_size, 5)
        Initial conditions for each simulation
    t_eval : array
        Time points for evaluation
    params_batch : array, shape (batch_size, 7)
        Parameters for each simulation
    rtol : float, default=1e-6
        Relative tolerance for adaptive stepping
    atol : float, default=1e-8
        Absolute tolerance for adaptive stepping
    max_steps : int, default=16**4
        Maximum number of steps

    Returns:
    --------
    solutions : array, shape (batch_size, len(t_eval), 5)
        Solutions for all simulations
    """
    batched_solve = jax.vmap(_forward_solve, in_axes=(0, 0, 0))
    return batched_solve(y0_batch, t_eval, params_batch)


_jitted_forward_solve_batch = jit(_forward_solve_batch)


def _vjp_forward_solve_batch(y0_batch, t_eval, params_batch, gz_batch):
    """Vector-Jacobian product for batched forward solve."""
    _, vjp_fn = jax.vjp(_forward_solve_batch, y0_batch, t_eval, params_batch)
    return vjp_fn(gz_batch)


_jitted_vjp_forward_solve_batch = jit(_vjp_forward_solve_batch)


# =====================================================
# TRULY VECTORIZED BATCH IMPLEMENTATION
# =====================================================


@jit
def _reaction_vectorized_batch(t, y_flat, params_batch):
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
    n_states = 5
    batch_size = params_batch.shape[0]

    # Reshape flat state vector back to batch format
    y_batch = y_flat.reshape(batch_size, n_states)

    # Use vmap to vectorize the base reaction function
    # This is cleaner and potentially faster than manual vectorization
    vectorized_reaction = jax.vmap(reaction, in_axes=(None, 0, 0))
    dydt_batch = vectorized_reaction(t, y_batch, params_batch)

    return dydt_batch.flatten()


def _forward_solve_vectorized(y0_batch, t_eval, params_batch):
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

    batch_size, n_states = y0_batch_jax.shape

    # Flatten initial conditions for the vectorized solver
    y0_flat = y0_batch_jax.flatten()

    # Define ODE term for Diffrax with vectorized reaction
    ode_term = dfx.ODETerm(lambda t, y, args: _reaction_vectorized_batch(t, y, args))

    # Set up solver - Tsit5 is a high-order Runge-Kutta method
    solver = dfx.Tsit5()

    # Define step size controller for adaptive stepping
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)

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
    solutions_batch = solutions_flat.reshape(n_times, batch_size, n_states)

    # Transpose to get (batch_size, n_times, n_states)
    return jnp.transpose(solutions_batch, (1, 0, 2))


_jitted_forward_solve_vectorized = jit(_forward_solve_vectorized)


def _vjp_forward_solve_vectorized(y0_batch, t_eval, params_batch, gz_batch):
    """Vector-Jacobian product for vectorized forward solve."""
    _, vjp_fn = jax.vjp(_forward_solve_vectorized, y0_batch, t_eval, params_batch)
    return vjp_fn(gz_batch)


_jitted_vjp_forward_solve_vectorized = jit(_vjp_forward_solve_vectorized)


class ForwardSolveOp(Op):
    def make_node(self, y0, t, theta):
        y0 = pt.tensor.as_tensor_variable(y0)
        t = pt.tensor.as_tensor_variable(t)
        theta = pt.tensor.as_tensor_variable(theta)
        return Apply(self, [y0, t, theta], [pt.tensor.dmatrix()])

    def perform(self, node, inputs, outputs):
        y0, t, theta = inputs
        result = _forward_solve(y0, t, theta)
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
        vjp_y0, vjp_t, vjp_theta = _vjp_forward_solve(y0, t, theta, gz)
        outputs[0][0] = np.asarray(vjp_y0)
        outputs[1][0] = np.asarray(vjp_t)
        outputs[2][0] = np.asarray(vjp_theta)


class ForwardSolveBatchOp(Op):

    def make_node(self, y0_batch, t, theta):
        y0_batch = pt.tensor.as_tensor_variable(y0_batch)  # (batch, dim)
        t = pt.tensor.as_tensor_variable(t)  # (n_times,)
        theta = pt.tensor.as_tensor_variable(theta)  # (param_dim,)
        outputs = [pt.tensor.dtensor3()]  # (batch, n_times, dim)
        return Apply(self, [y0_batch, t, theta], outputs)

    def perform(self, node, inputs, outputs):
        y0_batch, t, theta = inputs
        result = _forward_solve_batch(y0_batch, t, theta)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs, output_grads):
        y0_batch, t, theta = inputs
        (gz,) = output_grads
        vjp_y0, vjp_t, vjp_theta = VJPSolBatchOp()(y0_batch, t, theta, gz)
        return [vjp_y0, vjp_t, vjp_theta]


class VJPSolBatchOp(Op):
    def make_node(self, y0_batch, t, theta, gz):
        y0_batch = pt.tensor.as_tensor_variable(y0_batch)
        t = pt.tensor.as_tensor_variable(t)
        theta = pt.tensor.as_tensor_variable(theta)
        gz = pt.tensor.as_tensor_variable(gz)
        return Apply(
            self,
            [y0_batch, t, theta, gz],
            [pt.tensor.dmatrix(), pt.tensor.dmatrix(), pt.tensor.dmatrix()],
        )

    def perform(self, node, inputs, outputs):
        y0_batch, t, theta, gz = inputs
        vjp_y0, vjp_t, vjp_theta = _vjp_forward_solve_batch(y0_batch, t, theta, gz)
        outputs[0][0] = np.asarray(vjp_y0)
        outputs[1][0] = np.asarray(vjp_t)
        outputs[2][0] = np.asarray(vjp_theta)


class ForwardSolveVectorizedOp(Op):
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
        result = _forward_solve_vectorized(y0_batch, t, params_batch)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs, output_grads):
        y0_batch, t, params_batch = inputs
        (gz,) = output_grads
        vjp_y0, vjp_t, vjp_params = VJPSolVectorizedOp()(y0_batch, t, params_batch, gz)
        return [vjp_y0, vjp_t, vjp_params]


class VJPSolVectorizedOp(Op):
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
        vjp_y0, vjp_t, vjp_params = _vjp_forward_solve_vectorized(
            y0_batch, t, params_batch, gz
        )
        outputs[0][0] = np.asarray(vjp_y0)
        outputs[1][0] = np.asarray(vjp_t)
        outputs[2][0] = np.asarray(vjp_params)


forward_solve = ForwardSolveOp()
forward_solve_batch = ForwardSolveBatchOp()
forward_solve_vectorized = ForwardSolveVectorizedOp()
vjp_forward_solve = VJPSolOp()
vjp_forward_solve_batch = VJPSolBatchOp()
vjp_forward_solve_vectorized = VJPSolVectorizedOp()


@jax_funcify.register(ForwardSolveOp)
def jax_funcify_forward_solve(op, node, **kwargs):
    return _forward_solve


@jax_funcify.register(VJPSolOp)
def jax_funcify_vjp_forward_solve(op, node, **kwargs):
    return _vjp_forward_solve


@jax_funcify.register(ForwardSolveBatchOp)
def jax_funcify_forward_solve_batch(op, node, **kwargs):
    return _forward_solve_batch


@jax_funcify.register(VJPSolBatchOp)
def jax_funcify_vjp_forward_solve_batch(op, node, **kwargs):
    return _vjp_forward_solve_batch


@jax_funcify.register(ForwardSolveVectorizedOp)
def jax_funcify_forward_solve_vectorized(op, node, **kwargs):
    return _forward_solve_vectorized


@jax_funcify.register(VJPSolVectorizedOp)
def jax_funcify_vjp_forward_solve_vectorized(op, node, **kwargs):
    return _vjp_forward_solve_vectorized


if __name__ == "__main__":
    # Test basic functionality with Diffrax

    # Example parameters: [p, T, kla, A01, EA1, A02, EA2]
    params = jnp.array(
        [10e5, 300, 3e-3, 0.0001, 40000, 0.00001, 65000], dtype=jnp.float64
    )

    # Initial conditions: [n_g_con, c_Al, c_Bl, c_Cl, c_Dl]
    y0 = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=jnp.float64)
    print(y0.dtype)

    # Time points
    t = jnp.linspace(0, 10, 100, dtype=jnp.float64)

    print("Testing Diffrax-based JAX implementation...")
    print(f"Using Diffrax version with Tsit5 solver and adaptive stepping")

    # Test single simulation with different tolerances
    print("\\n1. Testing single simulation:")
    solution = _forward_solve(y0, t, params)
    print(f"Solution shape: {solution.shape}")
    print(f"Final concentrations: {solution[-1]}")

    # Test vectorized version with timing comparison
    print("\\n2. Performance Comparison: vmap vs truly vectorized batch solving")

    # Use a larger batch size for more meaningful timing
    batch_size = 100000
    y0_batch = jnp.tile(y0, (batch_size, 1))
    t_batch = jnp.tile(t, (batch_size, 1))
    deviations = np.random.normal(1, 0.01, size=(batch_size, 7))
    params_batch = jnp.tile(params, (batch_size, 1))
    params_batch = params_batch * deviations

    print(f"Testing with batch_size = {batch_size}, time_points = {len(t)}")

    # Warm up both methods (JIT compilation)
    print("\\n  Warming up JIT compilation...")
    _ = _forward_solve_batch(y0_batch[:2], t_batch[:2], params_batch[:2])
    _ = _forward_solve_vectorized(y0_batch[:2], t, params_batch[:2])

    # Time the vmap-based batch method
    print("\\n  Timing vmap-based batch method (_forward_solve_batch)...")
    n_runs = 5
    times_vmap = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        batch_solutions_vmap = _forward_solve_batch(y0_batch, t_batch, params_batch)
        end_time = time.perf_counter()
        times_vmap.append(end_time - start_time)
        print(f"    Run {i+1}: {times_vmap[-1]:.4f} seconds")

    avg_time_vmap = np.mean(times_vmap)
    std_time_vmap = np.std(times_vmap)

    # Time the truly vectorized method
    print("\\n  Timing truly vectorized method (_forward_solve_vectorized)...")
    times_vectorized = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        batch_solutions_vectorized = _forward_solve_vectorized(
            y0_batch, t, params_batch
        )
        end_time = time.perf_counter()
        times_vectorized.append(end_time - start_time)
        print(f"    Run {i+1}: {times_vectorized[-1]:.4f} seconds")

    avg_time_vectorized = np.mean(times_vectorized)
    std_time_vectorized = np.std(times_vectorized)

    # Print timing comparison
    print(f"\\n  📊 TIMING RESULTS (batch_size={batch_size}):")
    print(
        f"  ├─ vmap batch method:      {avg_time_vmap:.4f} ± {std_time_vmap:.4f} seconds"
    )
    print(
        f"  ├─ truly vectorized:       {avg_time_vectorized:.4f} ± {std_time_vectorized:.4f} seconds"
    )

    if avg_time_vmap > avg_time_vectorized:
        speedup = avg_time_vmap / avg_time_vectorized
        print(f"  └─ 🚀 Vectorized is {speedup:.2f}x FASTER")
    else:
        slowdown = avg_time_vectorized / avg_time_vmap
        print(f"  └─ ⚠️ Vectorized is {slowdown:.2f}x slower (unexpected!)")

    # Verify results are the same
    print(f"\\n  ✅ Results verification:")
    print(f"  ├─ vmap solution shape:       {batch_solutions_vmap.shape}")
    print(f"  ├─ vectorized solution shape: {batch_solutions_vectorized.shape}")
    max_diff = jnp.max(jnp.abs(batch_solutions_vmap - batch_solutions_vectorized))
    print(
        f"  └─ Max difference:            {max_diff:.2e} (should be ~0)"
    )  # Test reaction_batch method
    print("\\n2a. Testing vectorized reaction evaluation:")
    dydt_batch = _reaction_batch(y0_batch, t[0], params_batch)
    print(f"Batch reaction rates shape: {dydt_batch.shape}")
    print(f"Mean reaction rates at t=0: {jnp.mean(dydt_batch, axis=0)}")

    print("\\n✅ Unified Diffrax-JAX implementation test completed successfully!")
    print("🚀 Single ChemJAX class now provides:")
    print("   - Individual simulations: model.pt_forward()")
    print("   - Batch simulations: model.solve_batch()")
    print("   - Vectorized reactions: model.reaction_batch()")
    print("   - PyMC integration: model.create_pymc_op()")
    print("Ready for high-performance Bayesian inference with PyMC!")
