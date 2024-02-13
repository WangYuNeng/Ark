
from ark.simulate.dynamical_system import DynamicalSystem
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dr
import numpy as np

from ark.specification.attribute_def import AttrDef

class CircuitSystem(eqx.Module):
    """
    Circuit system with parameter values that can be sampled from.

    Running this multicore on CPU:
    The input dimension to be parallelized over must be sharded in the appropriate dimension (this is also true for multi-GPU).
    Also remember to set the XLA compile flags correctly (which is done by ark.simulate.utils.jax_settings when specifying the device).
    """

    dynamical_system: DynamicalSystem = eqx.field(static=True)
    "Dynamical system for simulation and parameter metadata"

    parameter_state: jax.Array
    "Current parameters"

    def clamp_parameters(self):
        state = self.parameter_state
        parameter_symbols = self.dynamical_system.parameter_symbols()

        # Create parameter minimum and maximum arrays
        def get_min_max(attr_def: AttrDef) -> tuple[float, float]:
            valid_range = attr_def.valid_range
            if valid_range is None:
                return -jnp.inf, jnp.inf
            if (exact := valid_range.exact) is not None:
                return exact, exact
            min_val = valid_range.min if valid_range.min is not None else -jnp.inf
            max_val = valid_range.max if valid_range.max is not None else jnp.inf
            return min_val, max_val

        min_max = [get_min_max(v.variability) for k, v in parameter_symbols.items()]
        clamp_array = np.array(min_max).T
        print(f"{clamp_array.shape=}")
        print(clamp_array)

        new_state = jnp.clip(state, clamp_array[0], clamp_array[1])

        return eqx.tree_at(lambda t: t.parameter_state, self, replace=new_state)

    def simulate(self, saveat: dr.SaveAt = dr.SaveAt(t1=True)) -> dr.Solution:
        solution = self.dynamical_system.solve_system(
            initial_values=jnp.zeros(
                len(self.dynamical_system.transient_symbols()), dtype=float
            ),
            parameter_values=self.parameter_state,
            saveat=saveat,
        )
        return solution

    def simulate_with_sensitivity(self, saveat = dr.SaveAt(t1=True)) -> dr.Solution:
        solution = self.dynamical_system.solve_system_with_sensitivity(
            initial_values=jnp.zeros(
                len(self.dynamical_system.transient_symbols()), dtype=float
            ),
            parameter_values=self.parameter_state,
            saveat=saveat,
        )
        return solution

    def sample_solve(
        self, key=jax.Array, saveat: dr.SaveAt = dr.SaveAt(t1=True)
    ) -> dr.Solution:
        sampled_parameters = self.dynamical_system.sample_parameters(
            nominal_values=self.parameter_state, key=key
        )
        manufactured_cs = CircuitSystem(
            dynamical_system=self.dynamical_system, parameter_state=sampled_parameters
        )
        return manufactured_cs.simulate(saveat)

    def sample_solve_with_sensitivity(
        self, key=jax.Array, saveat: dr.SaveAt = dr.SaveAt(t0=False, t1=True)
    ) -> dr.Solution:
        sampled_parameters = self.dynamical_system.sample_parameters(
            nominal_values=self.parameter_state, key=key
        )
        manufactured_cs = CircuitSystem(
            dynamical_system=self.dynamical_system, parameter_state=sampled_parameters
        )
        return manufactured_cs.simulate_with_sensitivity(saveat)

    def multi_solve(
        self,
        num_samples: int,
        key: jax.Array = jrandom.PRNGKey(0),
        saveat: dr.SaveAt = dr.SaveAt(t1=True),
    ) -> dr.Solution:
        # Create a set of seeds to vectorize over
        # n_devices = len(jax.devices())
        rngs = jrandom.bits(key=key, shape=(num_samples, 2), dtype=jnp.uint32)
        rngs = jax.device_put(
            rngs, jax.sharding.PositionalSharding(jax.devices()).reshape((-1, 1))
        )

        return eqx.filter_vmap(CircuitSystem.sample_solve, in_axes=(None, 0, None))(
            self, rngs, saveat
        )