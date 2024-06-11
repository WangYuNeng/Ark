from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax


@dataclass
class TimeInfo:

    t0: jax.typing.DTypeLike
    t1: jax.typing.DTypeLike
    dt0: jax.typing.DTypeLike
    saveat: list[jax.typing.DTypeLike]


class BaseAnalogCkt(eqx.Module):
    """Base class for differentiable analog circuit simulation.

    Attributes:
        trainable: The trainable parameters of the circuit.
        t0: The start time of the simulation.
        t1: The end time of the simulation.
        dt0: The time step.
        y0: The initial values of the state variables.
        saveat: The time points to save the simulation results.
        is_stochastic: Whether the simulation is stochastic (consider noise).
        solver: The ODE solver to use.
    """

    a_trainable: jax.Array
    d_trainable: list[jax.Array]
    is_stochastic: bool
    solver: diffrax.AbstractSolver

    # Future todo: Make time info and initial state trainable if needed

    def __init__(
        self,
        init_trainable: tuple[jax.Array, list[jax.Array]] | jax.Array,
        is_stochastic: bool,
        solver: diffrax.AbstractSolver,
    ) -> None:
        if isinstance(init_trainable, tuple):
            self.a_trainable, self.d_trainable = init_trainable
        elif isinstance(init_trainable, jax.Array):
            self.a_trainable = init_trainable
            self.d_trainable = []
        self.is_stochastic = is_stochastic
        self.solver = solver

    def __call__(
        self,
        time_info: TimeInfo,
        initial_state: jax.Array,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
        noise_seed: jax.typing.DTypeLike,
        gumble_temp: jax.typing.DTypeLike = 1,
    ):
        """The differentiable forward pass of the circuit simulation.

        Args:
            switch: The switch values for the circuit if any.
            mismatch_seed: The seed for the static random mismatch.
            noise_seed: The seed for the transient noise.
            solver: The ODE solver to use.
        """
        args = self.make_args(switch, mismatch_seed, gumble_temp)

        if not self.is_stochastic:
            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.ode_fn),
                solver=self.solver,
                t0=time_info.t0,
                t1=time_info.t1,
                dt0=time_info.dt0,
                y0=initial_state,
                saveat=diffrax.SaveAt(ts=time_info.saveat),
                args=args,
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
                args=args,
            )

        return self.readout(y=solution.ys)

    def make_args(
        self,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
    ) -> jax.Array:
        """Make the arguments for the ODE function."""

        raise NotImplementedError

    def ode_fn(
        self, t: jax.typing.DTypeLike, y: jax.Array, args: jax.Array
    ) -> jax.Array:
        """The ODE function to be solved."""

        raise NotImplementedError

    def noise_fn(
        self, t: jax.typing.DTypeLike, y: jax.Array, args: jax.Array
    ) -> jax.Array:
        """The noise function to be added to the ODE."""

        raise NotImplementedError

    def readout(self, y: jax.Array) -> jax.Array:
        """Readout the output of the circuit."""

        raise NotImplementedError

    @staticmethod
    def cdg_to_initial_states(cdg) -> list[jax.typing.DTypeLike]:
        """Extract the initial states from a CDG."""

        raise NotImplementedError

    @staticmethod
    def cdg_to_switch_array(cdg) -> list[int]:
        """Extract the switch values from a CDG."""

        raise NotImplementedError
