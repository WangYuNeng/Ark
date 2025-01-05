"""Execute Oscillator-Based Computing (OBC) in matrix form to resolve 
long compilation times for large networks."""

from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ark.optimization.base_module import TimeInfo


class OBCStateFunc(eqx.Module):
    coupling_aux: jax.Array  # A + A^T = coupling
    locking: jax.typing.DTypeLike
    coupling_mask: np.ndarray
    n_row: int
    n_col: int

    def __init__(
        self,
        init_coupling: jax.Array,
        init_locking: float,
        neighbor_connection: bool,
        **kwargs
    ):
        """Oscillator-based computing state update function.

        Args:
            init_coupling (jax.Array): coupling between any two oscillators (shape: N_COL x N_ROW x N_COL x N_ROW)
            init_locking (float): initial locking strength
            neighbor_connection (bool): whether force the connection to be only between neighbors
        """
        n_row, n_col = init_coupling.shape[0], init_coupling.shape[1]
        self.n_row, self.n_col = n_row, n_col
        init_coupling = init_coupling.reshape(n_row * n_col, n_row * n_col)
        assert (init_coupling == init_coupling.T).all()
        super().__init__(**kwargs)
        self.coupling_aux = init_coupling / 2
        self.locking = init_locking
        if not neighbor_connection:
            # All-to-all connection, no self-connection (considered in the injection locking term)
            self.coupling_mask = np.ones_like(init_coupling) - np.eye(
                init_coupling.shape[0]
            )
        else:
            # Neighbor connection, no self-connection (considered in the injection locking term)
            self.coupling_mask = np.zeros_like(init_coupling)

            for node_id in range(init_coupling.shape[0]):
                for node_id_ in range(init_coupling.shape[1]):
                    row, col = node_id // n_col, node_id % n_col
                    row_, col_ = node_id_ // n_col, node_id_ % n_col
                    if abs(row - row_) + abs(col - col_) == 1:
                        self.coupling_mask[node_id, node_id_] = 1
                        self.coupling_mask[node_id_, node_id] = 1

    def __call__(self, t, y: jax.Array, args):
        """The derivative function for the OBC state update."""
        coupling = self.coupling_aux + self.coupling_aux.T
        state_horizontal_stack = y.repeat(y.shape[0], axis=0).reshape(-1, y.shape[0])
        state_diff = state_horizontal_stack - state_horizontal_stack.T

        tot_coupling_strength = jnp.sum(
            jax.lax.mul(
                jax.lax.mul(jnp.array(self.coupling_mask), coupling),
                jnp.sin(jnp.pi * state_diff),
            ),
            axis=1,
        )

        lock_strength = self.locking * jnp.sin(2 * jnp.pi * y)
        return -tot_coupling_strength - lock_strength

    @property
    def coupling_weight(self):
        return (self.coupling_aux + self.coupling_aux.T).reshape(
            self.n_row, self.n_col, self.n_row, self.n_col
        )

    @property
    def locking_weight(self):
        return self.locking


class OscillatorNetworkMatrixSolve(eqx.Module):

    ode_fn: OBCStateFunc
    is_stochastic: bool
    solver: diffrax.AbstractSolver
    noise_fn: Callable

    def __init__(
        self,
        init_coupling: jax.Array,
        init_locking: float,
        neighbor_connection: bool,
        is_stochastic: bool,
        noise_amp: float,
        solver: diffrax.AbstractSolver,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ode_fn = OBCStateFunc(init_coupling, init_locking, neighbor_connection)
        self.is_stochastic = is_stochastic
        self.solver = solver
        self.noise_fn = lambda t, y, args: jnp.ones_like(y.shape[0]) * noise_amp

    def __call__(
        self,
        time_info: TimeInfo,
        initial_state: jax.Array,
        switch: jax.Array,  # Unused, for interface compatibility
        args_seed: jax.typing.DTypeLike,  # Unused, for interface compatibility
        noise_seed: jax.typing.DTypeLike,
        gumbel_temp: jax.typing.DTypeLike = 1,  # Unused, for interface compatibility
        hard_gumbel: bool = False,  # Unused, for interface compatibility
    ):
        if not self.is_stochastic:
            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.ode_fn),
                solver=self.solver,
                t0=time_info.t0,
                t1=time_info.t1,
                dt0=time_info.dt0,
                y0=initial_state,
                saveat=diffrax.SaveAt(ts=time_info.saveat),
            )
        else:
            ode_term = diffrax.ODETerm(self.ode_fn)
            brownian = diffrax.VirtualBrownianTree(
                time_info.t0,
                time_info.t1,
                tol=time_info.dt0 / 2,
                shape=initial_state.shape,
                key=jax.random.PRNGKey(noise_seed),
            )
            brownian_term = diffrax.WeaklyDiagonalControlTerm(self.noise_fn, brownian)
            solution = diffrax.diffeqsolve(
                terms=diffrax.MultiTerm(ode_term, brownian_term),
                solver=self.solver,
                t0=time_info.t0,
                t1=time_info.t1,
                dt0=time_info.dt0,
                y0=initial_state,
                saveat=diffrax.SaveAt(ts=time_info.saveat),
            )

        return solution.ys

    def weights(self):
        return self.ode_fn.coupling_weight, self.ode_fn.locking_weight
