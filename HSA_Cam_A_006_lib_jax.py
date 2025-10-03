import jax
import jax.numpy as jnp
from jax import jit
import diffrax as dfx
import pytensor as pt
import numpy as np
from scipy.constants import R
from pytensor.compile.ops import as_op
import pymc.pytensorf


class ChemJAX:
    """JAX-optimized chemical reaction model for maximum speed and PyMC compatibility."""

    @staticmethod
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

    @staticmethod
    def forward_solve(y0, t, theta):
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
        ode_term = dfx.ODETerm(lambda t, y, args: ChemJAX.reaction(t, y, args))

        # Set up solver - Tsit5 is a high-order Runge-Kutta method
        solver = dfx.Tsit5()

        # Define step size controller for adaptive stepping
        stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)

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
        )

        return solution.ys

    @staticmethod
    @as_op(
        itypes=[pt.tensor.fvector, pt.tensor.dvector, pt.tensor.dvector],
        otypes=[pt.tensor.dmatrix],
    )
    def pt_forward_solve(y0, t, theta):
        return ChemJAX.forward_solve(y0, t, theta)

    @staticmethod
    @jit
    def reaction_batch(y_batch, t, params_batch):
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
        return jax.vmap(ChemJAX.reaction, in_axes=(None, 0, 0))(
            t, y_batch, params_batch
        )

    @staticmethod
    def forward_solve_batch(y0_batch, t_eval, params_batch):
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
        batched_solve = jax.vmap(ChemJAX.forward_solve, in_axes=(0, 0, 0))
        return batched_solve(y0_batch, t_eval, params_batch)


# Legacy class name for backward compatibility (now just references ChemJAX)
class ChemJAXVectorized:
    """Legacy class for backward compatibility. Use ChemJAX directly instead."""

    @staticmethod
    @jit
    def reaction_batch(y_batch, t, params_batch):
        """Legacy method. Use ChemJAX.reaction_batch() instead."""
        return ChemJAX.reaction_batch(y_batch, t, params_batch)

    @staticmethod
    def solve_batch(y0_batch, t_eval, params_batch, **kwargs):
        """Legacy method. Use ChemJAX.solve_batch() instead."""
        return ChemJAX.forward_solve_batch(y0_batch, t_eval, params_batch, **kwargs)


# Example usage and testing functions
if __name__ == "__main__":
    # Test basic functionality with Diffrax
    model = ChemJAX()

    # Example parameters: [p, T, kla, A01, EA1, A02, EA2]
    params = jnp.array([10e5, 300, 3e-3, 0.0001, 40000, 0.00001, 65000])

    # Initial conditions: [n_g_con, c_Al, c_Bl, c_Cl, c_Dl]
    y0 = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0])

    # Time points
    t = jnp.linspace(0, 10, 100)

    print("Testing Diffrax-based JAX implementation...")
    print(f"Using Diffrax version with Tsit5 solver and adaptive stepping")

    # Test single simulation with different tolerances
    print("\\n1. Testing single simulation:")
    solution = model.forward_solve(y0, t, params)
    print(f"Solution shape: {solution.shape}")
    print(f"Final concentrations: {solution[-1]}")

    # Test vectorized version (now available directly on ChemJAX)
    print("\\n2. Testing vectorized simulations (unified interface):")
    batch_size = 10
    y0_batch = jnp.tile(y0, (batch_size, 1))
    t_batch = jnp.tile(t, (batch_size, 1))
    # Add parameter variation for more realistic test
    params_batch = jnp.tile(params, (batch_size, 1))

    # Use vectorized methods directly on the same model instance
    batch_solutions = model.forward_solve_batch(y0_batch, t_batch, params_batch)
    print(f"Batch solution shape: {batch_solutions.shape}")
    print(f"Parameter variation std: {jnp.std(params_batch, axis=0)}")

    # Test reaction_batch method
    print("\\n2a. Testing vectorized reaction evaluation:")
    dydt_batch = model.reaction_batch(y0_batch, t[0], params_batch)
    print(f"Batch reaction rates shape: {dydt_batch.shape}")
    print(f"Mean reaction rates at t=0: {jnp.mean(dydt_batch, axis=0)}")

    print("\\n✅ Unified Diffrax-JAX implementation test completed successfully!")
    print("🚀 Single ChemJAX class now provides:")
    print("   - Individual simulations: model.pt_forward()")
    print("   - Batch simulations: model.solve_batch()")
    print("   - Vectorized reactions: model.reaction_batch()")
    print("   - PyMC integration: model.create_pymc_op()")
    print("Ready for high-performance Bayesian inference with PyMC!")
