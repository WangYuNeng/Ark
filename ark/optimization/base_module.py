import diffrax
import equinox as eqx
import jax


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

    trainable: jax.Array
    t0: jax.typing.DTypeLike
    t1: jax.typing.DTypeLike
    dt0: jax.typing.DTypeLike
    y0: jax.Array  # Need option to set this to fixed or trainable
    saveat: list[jax.typing.DTypeLike]
    is_stochastic: bool
    solver: diffrax.AbstractSolver

    def __init__(
        self,
        init_trainable: jax.Array,
        is_stochastic: bool,
        solver: diffrax.AbstractSolver,
        **args
    ) -> None:
        self.trainable = init_trainable
        self.is_stochastic = is_stochastic
        self.solver = solver
        self.configure_simulation()
        for key, value in args:
            if key in self.__dict__:  # Check if the attribute exists
                setattr(self, key, value)

    def __call__(
        self,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
        noise_seed: jax.typing.DTypeLike,
    ):
        """The differentiable forward pass of the circuit simulation.

        Args:
            switch: The switch values for the circuit if any.
            mismatch_seed: The seed for the static random mismatch.
            noise_seed: The seed for the transient noise.
            solver: The ODE solver to use.
        """
        args = self.make_args(switch, mismatch_seed)

        if not self.is_stochastic:
            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.ode_fn),
                solver=self.solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=diffrax.SaveAt(ts=self.saveat),
                args=args,
            )
            return solution.ys
        else:
            ode_term = diffrax.ODETerm(self.ode_fn)
            brownian = diffrax.VirtualBrownianTree(
                self.t0,
                self.t1,
                tol=self.dt0 / 2,
                shape=self.y0.shape,
                key=jax.random.PRNGKey(noise_seed),
            )
            brownian_term = diffrax.ControlTerm(self.noise_fn, brownian)
            solution = diffrax.diffeqsolve(
                terms=diffrax.MultiTerm(ode_term, brownian_term),
                solver=self.solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=diffrax.SaveAt(ts=self.saveat),
                args=args,
            )
            return solution.ys

    def make_args(
        self,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
    ) -> jax.Array:
        """Make the arguments for the ODE function."""

        raise NotImplementedError

    def configure_simulation(self):
        """Set up the simulation timing information."""

        raise NotImplementedError

    def rescale_params(self, params: jax.Array) -> jax.Array:
        """Scale the trainable parameters from [-1, 1] to their physical range."""

        raise NotImplementedError

    def map_params(self, params: jax.Array) -> jax.Array:
        """Perform parameter transformations, e.g., table lookup, discretization,
        etc.

        """

        raise NotImplementedError

    def clip_params(self, params: jax.Array) -> jax.Array:
        """Clip the parameters to their physical range."""

        raise NotImplementedError

    def add_mismatch(self, params: jax.Array, seed: jax.typing.DTypeLike) -> jax.Array:
        """Add random mismatch to the parameters."""

        raise NotImplementedError

    def combine_args(self, params: jax.Array, switch: jax.Array) -> jax.Array:
        """Combine trainable parameters with switch values"""

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
