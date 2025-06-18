import jax
import jax.numpy as jnp
from sat_utils import BLUE_PHASE, FALSE_PHASE, TRUE_PHASE

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
    # Get the output of the model, the first n_var output is a 1D array of shape (2n,) representing
    # the phase values of -var[0], var[0], -var[1], var[1], -var[2], var[2], ..., -var[n], var[n]
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


def system_energy_loss(
    model: BaseAnalogCkt,
    init_states: jax.Array,
    switches: jax.Array,
    sol: jax.Array,
    adj_matrix: jax.Array,
    time_info: TimeInfo,
):
    "Calculate the oscillator system energy as the loss function."
    # Get the output of the model, the first n_var output is a 1D array of shape (2n,) representing
    # the phase values of -var[0], var[0], -var[1], var[1], -var[2], var[2], ..., -var[n], var[n]
    y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
        time_info, init_states, switches, 0, 0
    )

    # y_raw: (batch_size, 1, len(adj_matrix) - 3)
    # Squeeze y_raw to remove the second dimension
    y_raw = jnp.squeeze(y_raw, axis=1)  # Shape: (batch_size, len(adj_matrix) - 3)
    # Append the y_raw with (False, True, Blue) phase values to shape (batch_size, 1, len(adj_matrix))
    y_appended = jnp.concatenate(
        [
            y_raw,
            jnp.ones((y_raw.shape[0], 1)) * FALSE_PHASE,
            jnp.ones((y_raw.shape[0], 1)) * TRUE_PHASE,
            jnp.ones((y_raw.shape[0], 1)) * BLUE_PHASE,
        ],
        axis=-1,
    )

    # Take the pair-wise difference of y_appended
    y_diff = jnp.expand_dims(y_appended, axis=1) - jnp.expand_dims(
        y_appended, axis=2
    )  # Shape: (batch_size, len(adj_matrix), len(adj_matrix))

    # Calculate the energy as the adjacency matrix times the sum of cosine differences
    energy = -jnp.sum(
        adj_matrix * jnp.cos(jnp.pi * y_diff), axis=(1, 2)
    )  # Shape: (batch_size,)

    # Return the mean energy as the loss
    return jnp.mean(energy)
