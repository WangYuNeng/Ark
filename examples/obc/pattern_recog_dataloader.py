from typing import Callable

import jax.numpy as jnp
import numpy as np

from ark.cdg.cdg import CDG, CDGNode

# Create 5x3 arrays of the numbers 0-0
NUMBERS = {
    0: np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
    1: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]]),
    2: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]]),
    3: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]),
    4: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]]),
    5: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]]),
    6: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    7: np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1]]),
    8: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    9: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]),
}

# Node pattern: relative to the first node
NODE_PATTERNS = [None for _ in NUMBERS.keys()]
for i, pattern in NUMBERS.items():
    pattern_flat = pattern.flatten()
    NODE_PATTERNS[i] = jnp.abs(jnp.array(pattern_flat[1:] - pattern_flat[0]))


def dataloader(batch_size: int, n_node: int):
    """Data loader for many-to-many reconstruction task

    Goal: reconstruct any of the N_CLASS patterns from the random initial state
    Note: So far only work for N_CLASS = 1
    """
    while True:
        x_init_states = np.random.rand(batch_size, n_node)
        noise_seed = np.random.randint(0, 2**32 - 1, size=batch_size)
        yield jnp.array(x_init_states), jnp.array(noise_seed)


def dataloader2(
    batch_size: int,
    n_class: int,
    graph: CDG,
    osc_array: list[list[CDGNode]],
    mapping_fn: Callable,
    snp_prob: float,
    gauss_std: float,
):
    """Data loader for one-to-one reconstruction task

    Goal: reconstruct the original pattern from the noisy pattern

    The training set is pairs of (noisy_numbers, tran_noise_seed, pattern).
    the pattern is the corresponding oscillator phase that represents the number.

    Args:
        batch_size: The number of samples in a batch
        graph: The CDG of the circuit
        osc_array: The array of oscillators (N_ROW x N_COL)
        mapping_fn: The mapping function from cdg nodes to the initial state array
        snp_prob: The probability of salt-and-pepper noise
        gauss_std: The standard deviation of the Gaussian noise
    """

    while True:
        # Sample batch_size numbers from 0-N_CLASS
        sampled_numbers = np.random.choice([i for i in range(n_class)], size=batch_size)

        # Generate the noisy numbers
        x, y = [], []
        n_row, n_col = len(osc_array), len(osc_array[0])
        for number in sampled_numbers:
            node_init = NUMBERS[number].copy().astype(np.float64)
            ideal_pattern = NODE_PATTERNS[number]

            # Add salt-and-pepper noise
            snp_mask = np.random.rand(*node_init.shape) < snp_prob
            node_init[snp_mask] = 1 - node_init[snp_mask]

            # Add Gaussian noise
            node_init += np.random.normal(0, gauss_std, node_init.shape)

            # Assign the initial state to the nodes
            for row in range(n_row):
                for col in range(n_col):
                    osc_array[row][col].set_init_val(node_init[row, col], 0)

            x.append(mapping_fn(graph))
            y.append(ideal_pattern)

        x, y = jnp.array(x), jnp.array(y)

        noise_seed = jnp.array(np.random.randint(0, 2**32 - 1, size=batch_size))
        # print(f"loss: {periodic_mse(x, y):.4f}")
        yield x, noise_seed, y
