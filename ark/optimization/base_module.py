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
        a_trainable: Analog trainable parameters of the circuit.
        d_trainable: Discrete trainable parameters of the circuit.
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
        args_seed: jax.typing.DTypeLike,
        noise_seed: jax.typing.DTypeLike,
        gumbel_temp: jax.typing.DTypeLike = 1,
        hard_gumbel: bool = False,
        max_steps: int = 4096,
    ):
        """The differentiable forward pass of the circuit simulation.

        Args:
            time_info: The time information for the simulation, including the start time,
                end time, time step, and time points to save the solution.
            initial_state: The initial state of the state variables in the diffeq.
            switch: The switch values for the circuit if any.
            args_seed: The seed for the static randomness when intialization the arguments.
                incluing the guassian for mismatch and gumbel distribution for gumbel softmax.
            noise_seed: The seed for the transient noise.
            gumbel_temp: The temperature for the gumbel softmax.
            hard_gumbel: Whether to use the hard gumbel softmax and straight-through estimator
                or the soft gumbel softmax. Default is False.
        """
        args = self.make_args(switch, args_seed, gumbel_temp, hard_gumbel)

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
                max_steps=max_steps,
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
                max_steps=max_steps,
            )

        return self.readout(y=solution.ys)

    def make_args(
        self,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
        gumbel_temp: jax.typing.DTypeLike,
        hard_gumbel: bool,
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

    @staticmethod
    def node_to_init_state_id(node_name: str) -> int:
        """Map the node name to the initial state id."""

        raise NotImplementedError

    @staticmethod
    def switch_to_args_id(switch: int) -> int:
        """Map the switch value to the args id."""

        raise NotImplementedError

    def weights(self) -> tuple[jax.Array, list[jax.Array]]:
        """Return a copy of the trainable parameters of the circuit."""

        return (self.a_trainable.copy(), self.d_trainable.copy())
