from typing import Callable

import jax
import jax.numpy as jnp
from pattern_recog_dataloader import NODE_PATTERNS

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo


def raw_to_rel_phase(raw_phase_out: jax.Array):
    """Convert the raw phase to the relative phase"""
    n_repeat = raw_phase_out.shape[1] - 1
    rel_phase = raw_phase_out[:, 1:] - raw_phase_out[:, 0].repeat(n_repeat).reshape(
        -1, n_repeat
    )
    return rel_phase


def normalize_angular_diff(y_end_readout: jax.Array, y: jax.Array):
    x = raw_to_rel_phase(y_end_readout)
    return jnp.sin(jnp.pi * ((x - y) / 2 % 1))


def periodic_mse(y_end_readout: jax.Array, y: jax.Array):
    y_end_readout = y_end_readout % 2
    rel_y = jnp.abs(raw_to_rel_phase(y_end_readout))
    phase_diff = jnp.where(rel_y > 1, 2 - rel_y, rel_y)
    return jnp.mean(jnp.square(phase_diff - y))


def periodic_mean_max_se(y_end_readout: jax.Array, y: jax.Array):
    y_end_readout = y_end_readout % 2
    rel_y = jnp.abs(raw_to_rel_phase(y_end_readout))
    phase_diff = jnp.where(rel_y > 1, 2 - rel_y, rel_y)
    return jnp.mean(jnp.max(jnp.square(phase_diff - y), axis=1))


def min_rand_reconstruction_loss(
    model: BaseAnalogCkt,
    x: jax.Array,
    noise_seed: jax.Array,
    gumbel_temp: float,
    time_info: TimeInfo,
    diff_fn: Callable,
    N_CLASS: int,
):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0, None))(
        time_info, x, [], 0, noise_seed, gumbel_temp
    )
    y_end_readout = y_raw[:, -1, :]
    losses = []
    for i in range(N_CLASS):
        losses.append(jnp.mean(diff_fn(y_end_readout, NODE_PATTERNS[i])))
    losses = jnp.array(losses)
    # Return the minimum average loss
    return jnp.min(losses)


def pattern_reconstruction_loss(
    model: BaseAnalogCkt,
    x: jax.Array,
    noise_seed: jax.Array,
    y: jax.Array,
    gumbel_temp: float,
    time_info: TimeInfo,
    diff_fn: Callable,
):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0, None))(
        time_info, x, [], 0, noise_seed, gumbel_temp
    )
    return diff_fn(y_raw[:, -1, :], y)
