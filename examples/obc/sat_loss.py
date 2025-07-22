import interpax
import jax
import jax.numpy as jnp
from sat_utils import (
    BLUE_PHASE,
    FALSE_PHASE,
    TRUE_PHASE,
    Assignment,
    Problem,
    n_sat_clauses,
)

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo

phase_x = jnp.array([0, 2 / 3, 4 / 3, 2])
bool_value = jnp.array([-1, 1, 0, -1])

phase_to_bool_fit = interpax.Akima1DInterpolator(phase_x, bool_value, check=False)


def assignment_to_phase(assignment: jax.Array) -> jax.Array:
    """FALSE -> FALSE_PHASE, TRUE -> TRUE_PHASE"""
    return jnp.where(assignment, TRUE_PHASE, FALSE_PHASE)


def phase_to_bool_assignments(assignment_phase: jax.Array) -> list[bool]:
    """Convert phase values to variable assignments.

    Args:
        assignment_phase (jax.Array): Phase values of the variables, shape (n_vars,).

    Returns:
        list[bool]: List of boolean assignments.
    """
    # Variable is TRUE iff the phase is within TRUE_PHASE +/- (TRUE_PHASE - FALSE_PHASE) / 2
    modular_phase = jnp.mod(assignment_phase, 2.0)
    threshold = (TRUE_PHASE - FALSE_PHASE) / 2
    return jnp.array(
        [jnp.abs(phase - TRUE_PHASE) < threshold for phase in modular_phase]
    )


def phase_to_assignment_ste(assignment_phase: jax.Array) -> jax.Array:
    """Convert phase values to variable assignments with a straight-through estimator

    Args:
        assignment_phase (jax.Array): Phase values of the variables, shape (n_vars,).

    Returns:
        jax.Array: Array of variable assignments.
    """
    modular_phase = jnp.mod(assignment_phase, 2.0)

    assignment = jnp.where(
        modular_phase
        < (TRUE_PHASE + FALSE_PHASE) / 2,  # If phase is closer to FALSE_PHASE
        -1,
        jnp.where(
            modular_phase
            < (TRUE_PHASE + BLUE_PHASE) / 2,  # If phase is closer to TRUE_PHASE
            1,
            jnp.where(
                modular_phase
                < (BLUE_PHASE + 2) / 2,  # If phase is closer to BLUE_PHASE
                0,
                -1,  # If phase is closer to FALSE_PHASE again
            ),
        ),
    )

    assignment_approx = jnp.interp(modular_phase, phase_x, bool_value)

    # Use straight-through estimator to allow gradients to pass through
    assignment_ste = (
        jax.lax.stop_gradient(assignment)
        + assignment_approx
        - jax.lax.stop_gradient(assignment_approx)
    )

    return assignment_ste


def loss_w_sol(
    model: BaseAnalogCkt,
    init_states: jax.Array,
    switches: jax.Array,
    sol: jax.Array,
    adj_matrix: jax.Array,
    n_vars: int,
    problems: jax.Array,
    trasform_mats: jax.Array,
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
    y_raw = jnp.squeeze(y_raw, axis=1)  # Shape: (batch_size, len(adj_matrix) - 3)
    # FIXME: Modular is too strict. E.g., because phase is periodic, 1.9 is close to 0
    # and the loss should be small if the solution is 0.
    y_modular = jnp.mod(y_raw[:, :n_vars], 2.0).reshape(sol.shape)

    # Convert the solution assignment to phase values
    sine_sol = assignment_to_phase(sol)

    # Calculate the loss
    loss = jnp.mean((y_modular - sine_sol) ** 2)
    return loss, y_raw


def system_energy_loss(
    model: BaseAnalogCkt,
    init_states: jax.Array,
    switches: jax.Array,
    sol: jax.Array,
    adj_matrix: jax.Array,
    n_vars: int,
    problems: jax.Array,
    trasform_mats: jax.Array,
    time_info: TimeInfo,
) -> tuple[jax.Array, jax.Array]:
    """Calculate the oscillator system energy as the loss function.

    Return:
        tuple[jax.Array, jax.Array]: Mean energy of the system and the raw phase values.
    """
    # Get the output of the model, the first n_var output is a 1D array of shape (2n,) representing
    # the phase values of -var[0], var[0], -var[1], var[1], -var[2], var[2], ..., -var[n], var[n]
    y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
        time_info, init_states, switches, 0, 0
    )

    # y_raw: (batch_size, 1, len(adj_matrix) - 3)
    # Squeeze y_raw to remove the second dimension
    y_raw = jnp.squeeze(y_raw, axis=1)  # Shape: (batch_size, len(adj_matrix) - 3)

    energy = phase_to_energy(
        phase_raw=y_raw,
        adj_matrix=adj_matrix,
    )  # Shape: (batch_size,)

    # Return the mean energy as the loss, satisfied clauses ratio as a metric
    return jnp.mean(energy), y_raw


def approx_sat_loss(
    model: BaseAnalogCkt,
    init_states: jax.Array,
    switches: jax.Array,
    sol: jax.Array,
    adj_matrix: jax.Array,
    n_vars: int,
    problems: jax.Array,
    trasform_mats: jax.Array,
    time_info: TimeInfo,
) -> tuple[jax.Array, jax.Array]:
    """Calculate the oscillator system energy as the loss function.

    Return:
        tuple[jax.Array, jax.Array]: Mean energy of the system and the raw phase values.
    """
    # Get the output of the model, the first n_var output is a 1D array of shape (2n,) representing
    # the phase values of -var[0], var[0], -var[1], var[1], -var[2], var[2], ..., -var[n], var[n]
    y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
        time_info, init_states, switches, 0, 0
    )

    # y_raw: (batch_size, 1, len(adj_matrix) - 3)
    # Squeeze y_raw to remove the second dimension
    y_raw = jnp.squeeze(y_raw, axis=1)  # Shape: (batch_size, len(adj_matrix) - 3)

    sat_loss = phase_to_approx_sat_loss(
        n_vars=n_vars,
        phase_raw=y_raw,
        transform_mats=trasform_mats,
    )  # Shape: (batch_size,)

    # Return the mean energy as the loss, satisfied clauses ratio as a metric
    return jnp.mean(sat_loss), y_raw


def phase_to_energy(
    phase_raw: jax.Array,
    adj_matrix: jax.Array,
):
    # Append the y_raw with (False, True, Blue) phase values to shape (batch_size, 1, len(adj_matrix))
    phase_appended = jnp.concatenate(
        [
            phase_raw,
            jnp.ones((phase_raw.shape[0], 1)) * FALSE_PHASE,
            jnp.ones((phase_raw.shape[0], 1)) * TRUE_PHASE,
            jnp.ones((phase_raw.shape[0], 1)) * BLUE_PHASE,
        ],
        axis=-1,
    )

    # Take the pair-wise difference of y_appended
    y_diff = jnp.expand_dims(phase_appended, axis=1) - jnp.expand_dims(
        phase_appended, axis=2
    )  # Shape: (batch_size, len(adj_matrix), len(adj_matrix))

    # Calculate the energy as the adjacency matrix times the sum of cosine differences
    energy = -jnp.sum(
        adj_matrix * jnp.cos(jnp.pi * y_diff), axis=(1, 2)
    )  # Shape: (batch_size,)

    return energy


def phase_to_sat_clause_rate(
    n_vars: int,
    phase_raw: jax.Array,
    problems: list[Problem],
) -> jax.Array:
    """Calculate the number of satisfied clauses in the SAT problems.
    Args:
        n_vars (int): Number of variables in the SAT problem.
        phase_raw (jax.Array): Phase values of the variables, shape (n_problmes, n_oscs).
        problems (list[Problem]): List of SAT problems.

    Returns:
        jax.Array: Ratio of satisfied clauses.
    """
    # Take the POSITIVE oscillators' phases
    var_phases = phase_raw[:, 1 : n_vars * 2 : 2]

    # Map the variable phases to boolean assignments
    bool_assignments = jax.vmap(phase_to_bool_assignments, in_axes=0)(var_phases)

    # Calculate the number of satisfied clauses for each problem
    n_sat_clause_list = []
    for clauses, assignment in zip(problems, bool_assignments):
        # Count the number of satisfied clauses
        n_sat_clause_list.append(n_sat_clauses(clauses, assignment))

    # Convert the list to a jax array
    n_satisfied_clauses = jnp.array(n_sat_clause_list)

    # Calculate the ratio of satisfied clauses
    n_clauses = jnp.array([len(prob) for prob in problems])
    ratio_satisfied = n_satisfied_clauses / n_clauses
    return ratio_satisfied


def phase_to_approx_sat_loss(
    n_vars: int,
    phase_raw: jax.Array,
    transform_mats: jax.Array,
) -> jax.Array:
    """Calculate the differentiable SAT loss based on the phase values.

    Args:
        n_vars (int): Number of variables in the SAT problem.
        phase_raw (jax.Array): Phase values of the variables, shape (n_problems, n_oscs).
        transform_mats (jax.Array): Transformation matrices for the SAT problems, shape (n_problems (batch_size),
        n_clauses, 3, n_vars).

    Returns:
        jax.Array: Differentiable SAT loss.
    """
    # Take the POSITIVE oscillators' phases
    var_phases = phase_raw[:, 1 : n_vars * 2 : 2]
    var_bools = phase_to_assignment_ste(var_phases)  # Shape: (batch_size, n_vars)

    var_vals_in_clauses = jax.vmap(
        lambda var_bools, transform_mat: transform_mat @ var_bools, in_axes=(0, 0)
    )(
        var_bools, transform_mats
    )  # Shape: (batch_size, n_clauses, 3)

    # map [-1, 1] to [0, -] using softplus
    # This penalizes values around 0 so that phases synchronize to blue or
    # between true and false are penalized
    alpha = 0.1  # Small positive constant to avoid zero
    shifted_var_vals = (
        jax.nn.leaky_relu(var_vals_in_clauses, negative_slope=alpha) + alpha
    )

    # Sum up along the last dimension to get the clause values and clip
    clause_vals = jax.nn.tanh(
        jnp.sum(shifted_var_vals, axis=-1)
    )  # Shape: (batch_size, n_clauses)

    return -jnp.mean(clause_vals, axis=-1)  # Mean over clauses for each problem
