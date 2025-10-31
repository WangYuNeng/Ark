"""Execute Oscillator-Based Computing (OBC) in matrix form to resolve
long compilation times for large networks."""

from typing import Callable, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class OBCStateFunc(eqx.Module):

    grid_coupling: jax.Array
    input_coupling: jax.Array
    _locking: jax.Array
    input_mask: np.ndarray
    grid_mask: np.ndarray
    n_row: int = eqx.field(static=True)
    n_col: int = eqx.field(static=True)
    pos_locking: bool
    reference_coupling: Optional[jax.Array]

    def __init__(
        self,
        grid_coupling: jax.Array,
        input_coupling: jax.Array,
        init_locking: jax.Array,
        input_kernel_size: int,
        grid_kernel_size: int,
        reference_coupling: Optional[jax.Array] = None,
        pos_locking: bool = True,
        **kwargs,
    ):
        """Oscillator-based computing state update function.

        Args:
            grid_coupling (jax.Array): coupling between grid oscillators (shape: N_COL x N_ROW x N_COL x N_ROW)
            input_coupling (jax.Array): coupling from input to grid oscillator (shape: N_COL x N_ROW x N_COL x N_ROW)
            init_locking (float): initial locking strength for all oscillators (shape: N_COL x N_ROW)
            input_kernel_size (int): size of the input-to-grid coupling kernel (e.g., 3 for 3x3). If negative, no connection
                from input to the grid.
            grid_kernel_size (int): size of the grid-to-grid coupling kernel (e.g., 3 for 3x3)
            has_reference (bool): whether there is a reference oscillator with constant phase
        """
        assert (
            grid_coupling.shape == input_coupling.shape
        ), f"Grid and input coupling must have the same shape. Got {grid_coupling.shape} and {input_coupling.shape}."
        n_row, n_col = grid_coupling.shape[0], grid_coupling.shape[1]
        self.n_row, self.n_col = n_row, n_col
        self.grid_coupling = grid_coupling
        self.input_coupling = input_coupling

        assert init_locking.shape == (
            n_row,
            n_col,
        ), f"Locking shape ({init_locking.shape}) must match grid shape ({n_row}, {n_col})."
        self._locking = init_locking

        self.grid_mask = self.get_kernel_mask(grid_kernel_size)
        self.input_mask = self.get_kernel_mask(input_kernel_size)

        self.reference_coupling = reference_coupling
        self.pos_locking = pos_locking
        super().__init__(**kwargs)

    def __call__(self, t, y: jax.Array, args):
        """The derivative function for the OBC state update.

        Args:
            t: time
            y (jax.Array): current phases of the oscillators (shape: N_ROW * N_COL)
            args: dictionary containing additional arguments, including:
                - input_phase (jax.Array): phases of the input oscillators (shape: N_ROW * N_COL)
        """
        input_phase = args["input_phase"]
        grid_coupling = self.grid_coupling_matrix
        input_coupling = self.input_coupling_matrix

        phase_diff = y[:, None] - y[None, :]
        input_phase_diff = y[:, None] - input_phase[None, :]

        tot_coupling_strength = jnp.sum(
            jax.lax.mul(grid_coupling, jnp.sin(jnp.pi * phase_diff))
            + jax.lax.mul(input_coupling, jnp.sin(jnp.pi * input_phase_diff)),
            axis=1,
        )
        lock_strength = self.locking.flatten() * jnp.sin(2 * jnp.pi * y)

        dydt = tot_coupling_strength - lock_strength

        if self.reference_coupling is not None:
            ref_phase = 1.0
            ref_coupling_strength = self.reference_coupling.flatten() * jnp.sin(
                jnp.pi * (y - ref_phase)
            )
            dydt += ref_coupling_strength

        return dydt

    @property
    def grid_coupling_matrix(self):
        m = jax.lax.mul(self.grid_coupling, jnp.array(self.grid_mask)).reshape(
            self.n_osc, self.n_osc
        )
        m = m - jnp.diag(jnp.diag(m))  # No self-coupling
        return m

    @property
    def input_coupling_matrix(self):
        return jax.lax.mul(self.input_coupling, jnp.array(self.input_mask)).reshape(
            self.n_osc, self.n_osc
        )

    @property
    def locking(self):
        if self.pos_locking:
            return jnp.abs(self._locking)
        else:
            return self._locking

    @property
    def n_osc(self) -> int:
        return self.n_row * self.n_col

    def get_kernel_mask(self, kernel_size: int) -> list:
        """Generate a kernel mask for convolution-like coupling.

        Args:
            kernel_size (int): size of the kernel (e.g., 3 for 3x3  kernel)
        """
        if kernel_size < 0:
            return np.zeros((self.n_row, self.n_col, self.n_row, self.n_col))
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        pad = kernel_size // 2
        mask = np.zeros((self.n_row, self.n_col, self.n_row, self.n_col))
        for i in range(self.n_row):
            for j in range(self.n_col):
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        ni = i + ki
                        nj = j + kj
                        if 0 <= ni < self.n_row and 0 <= nj < self.n_col:
                            mask[i, j, ni, nj] = 1
        return mask


class OscillatorReservoir(eqx.Module):

    ode_fn: OBCStateFunc
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    save_at: list = eqx.field(static=True)

    def __init__(
        self,
        ode_fn: OBCStateFunc,
        solver: diffrax.AbstractSolver,
        save_at: list,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ode_fn = ode_fn
        self.solver = solver
        self.save_at = save_at

    def __call__(
        self,
        initial_state: jax.Array,
        args: dict,
        t_end: float,
    ):
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.ode_fn),
            solver=self.solver,
            args=args,
            t0=0,
            t1=t_end,
            dt0=0.01,
            y0=initial_state,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=self.save_at),
        )

        return solution.ys

    @property
    def n_osc(self) -> int:
        return self.ode_fn.n_osc

    @property
    def n_row(self) -> int:
        return self.ode_fn.n_row

    @property
    def n_col(self) -> int:
        return self.ode_fn.n_col


class ReservoirWithLinear(eqx.Module):

    reservoir: OscillatorReservoir
    linear: eqx.nn.Linear
    output_dim: int = eqx.field(static=True)
    downsample: int = eqx.field(static=True)

    def __init__(
        self,
        reservoir: OscillatorReservoir,
        output_dim: int,
        downsample: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reservoir = reservoir
        self.output_dim = output_dim
        self.downsample = downsample

        linear_input_dim = (
            (self.reservoir.n_row // downsample)
            * (self.reservoir.n_col // downsample)
            * len(self.reservoir.save_at)
        )
        self.linear = eqx.nn.Linear(
            in_features=linear_input_dim,
            out_features=output_dim,
            use_bias=False,
            key=jax.random.PRNGKey(np.random.randint(0, 2**31 - 1)),
        )

    def __call__(
        self,
        initial_state: jax.Array,
        input_phase: jax.Array,
        t_end: float,
    ):
        args = {"input_phase": input_phase}
        res_out: jax.Array = self.reservoir(
            initial_state=initial_state,
            args=args,
            t_end=t_end,
        )  # shape : (len(saveat), self.reservoir.n_osc)

        n_timepoint, _ = res_out.shape
        n_row, n_col = self.reservoir.n_row, self.reservoir.n_col
        res_out = res_out.reshape(n_timepoint, n_row, n_col)
        res_out = res_out[:, :: self.downsample, :: self.downsample].flatten()

        output = self.linear(res_out)
        return output


if __name__ == "__main__":

    # Test the kernel mask generation
    n_row, n_col = 3, 3
    grid_coupling = jnp.ones((n_row, n_col, n_row, n_col))
    input_coupling = jnp.ones((n_row, n_col, n_row, n_col))
    init_locking = jnp.ones((n_row, n_col)) * 0.5
    obc_func = OBCStateFunc(
        grid_coupling=grid_coupling,
        input_coupling=input_coupling,
        init_locking=init_locking,
        input_kernel_size=-1,
        grid_kernel_size=3,
    )
    print(obc_func.grid_coupling_matrix)
