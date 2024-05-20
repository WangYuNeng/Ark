import diffrax
import equinox as eqx
import jax


def ode_fn(t: jax.typing.DTypeLike, y: jax.Array, args: jax.Array) -> jax.Array:
    """The dynamical system equations of the analog circuit."""

    raise NotImplementedError


class BaseAnalogCkt(eqx.Module):
    """Base class for differentiable analog circuit simulation.

    Attributes:
        trainable: The trainable parameters of the circuit.
        t0: The start time of the simulation.
        t1: The end time of the simulation.
        dt0: The time step.
        y0: The initial values of the state variables.
        saveat: The time points to save the simulation results.
    """

    trainable: jax.Array
    t0: jax.typing.DTypeLike
    t1: jax.typing.DTypeLike
    dt0: jax.typing.DTypeLike
    y0: jax.Array  # Need option to set this to fixed or trainable
    saveat: list[jax.typing.DTypeLike]

    def __init__(self, init_vals, is_stochastic: bool) -> None:
        self.trainable = init_vals
        self.is_stochastic = is_stochastic
        self.configure_simulation()

    def __call__(
        self,
        switch: jax.Array,
        mismatch_seed: jax.typing.DTypeLike,
        noise_seed: jax.typing.DTypeLike,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    ):
        """The differentiable forward pass of the circuit simulation.

        Args:
            switch: The switch values for the circuit if any.
            mismatch_seed: The seed for the static random mismatch.
            noise_seed: The seed for the transient noise.
            solver: The ODE solver to use.
        """
        rescaled_params = self.rescale_params(self.trainable)
        mapped_params = self.map_params(rescaled_params)
        clipped_params = self.clip_params(mapped_params)
        # Perform mismatch after clipping. Could change order if needed.
        mismatched_params = self.add_mismatch(clipped_params, mismatch_seed)
        args = self.combine_args(mismatched_params, switch)

        if not self.is_stochastic:
            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(ode_fn),
                solver=solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=diffrax.SaveAt(ts=self.saveat),
                args=args,
            )
            return solution.ys
        else:
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
