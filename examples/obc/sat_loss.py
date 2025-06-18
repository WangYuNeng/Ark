import jax
import jax.numpy as jnp
from sat_utils import FALSE_PHASE, TRUE_PHASE

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo


def assignment_to_phase(assignment: jax.Array) -> jax.Array:
    """FALSE -> FALSE_PHASE, TRUE -> TRUE_PHASE"""
    return jnp.where(assignment, TRUE_PHASE, FALSE_PHASE)


def loss_w_sol(
    model: BaseAnalogCkt,
    init_states: jax.Array,
    switches: jax.Array,
    sol: jax.Array,
    adj_matrix: jax.Array,
    time_info: TimeInfo,
):
    """
    Loss function for the 3-SAT problem. The loss is the sum of the squared differences between
    the output of the model and the target value (1 for True, 0 for False).
    """
    bz, n_vars = sol.shape
    # Get the output of the model, the output is a 1D array of shape (2n,) representing the phase values
    # of -var[0], var[0], -var[1], var[1], -var[2], var[2], ..., -var[n], var[n]
    y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
        time_info, init_states, switches, 0, 0
    )
    # FIXME: Modular is too strict. E.g., because phase is periodic, 1.9 is close to 0
    # and the loss should be small if the solution is 0.
    y_modular = jnp.mod(y_raw[:, :, :n_vars], 2.0).reshape(sol.shape)

    # Convert the solution assignment to phase values
    sine_sol = assignment_to_phase(sol)

    # Calculate the loss
    loss = jnp.mean((y_modular - sine_sol) ** 2)
    return loss
