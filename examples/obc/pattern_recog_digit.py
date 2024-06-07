from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from diffrax.solver import Heun
from jaxtyping import PyTree
from xor import (
    Coupling,
    Osc_modified,
    T,
    TrainableMananger,
    coupling_fn,
    locking_fn,
    obc_spec,
)

from ark.cdg.cdg import CDG, CDGNode
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler

jax.config.update("jax_enable_x64", True)


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

N_ROW, N_COL = 5, 3
N_NODE = N_ROW * N_COL
N_EDGE = (N_ROW - 1) * N_COL + (N_COL - 1) * N_ROW
N_CLASS = 5

N_CYCLES = 1
N_TIME_POINT = 100

time_info = TimeInfo(
    t0=0,
    t1=T * N_CYCLES,
    dt0=T / 50,
    saveat=jnp.linspace(0, T * N_CYCLES, N_TIME_POINT, endpoint=True),
)

N_PLOT_EVOLVE = 5
PLOT_TIME = [int(N_TIME_POINT / (N_PLOT_EVOLVE - 1) * i) for i in range(N_PLOT_EVOLVE)]


LEARNING_RATE = 1e-1
optim = optax.adam(LEARNING_RATE)

STEPS = 32
BZ = 256
VALIDATION_BZ = 4096
PLOT_BZ = 4

TASK = "one-to-one"  # "rand-to-many"
SNP_PROB, GAUSS_STD = 0.0, 0.2

DIFF_FN = "periodic_mse"  # "normalize_angular_diff


def pattern_to_edge_initialization(pattern: np.ndarray):
    """Map the pattern to the edge initialization

    If two pixels are different, the edge is initialized with -1.0
    Otherwise, the edge is initialized with 1.0
    """

    edge_init = []

    for row in range(N_ROW):
        for col in range(N_COL - 1):
            diff = pattern[row, col] - pattern[row, col + 1]
            edge_init.append(-1.0 if diff else 1.0)

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            diff = pattern[row, col] - pattern[row + 1, col]
            edge_init.append(-1.0 if diff else 1.0)

    return np.array(edge_init)


def dataloader(batch_size: int):
    """Data loader for many-to-many reconstruction task

    Goal: reconstruct any of the N_CLASS patterns from the random initial state
    Note: So far only work for N_CLASS = 1
    """
    while True:
        x_init_states = np.random.rand(batch_size, N_NODE)
        noise_seed = np.random.randint(0, 2**32 - 1, size=batch_size)
        yield jnp.array(x_init_states), jnp.array(noise_seed)


def dataloader2(
    batch_size: int,
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
        sampled_numbers = np.random.choice([i for i in range(N_CLASS)], size=batch_size)

        # Generate the noisy numbers
        x, y = [], []
        for number in sampled_numbers:
            node_init = NUMBERS[number].copy().astype(np.float64)
            ideal_pattern = NODE_PATTERNS[number]

            # Add salt-and-pepper noise
            snp_mask = np.random.rand(*node_init.shape) < snp_prob
            node_init[snp_mask] = 1 - node_init[snp_mask]

            # Add Gaussian noise
            node_init += np.random.normal(0, gauss_std, node_init.shape)

            # Assign the initial state to the nodes
            for row in range(N_ROW):
                for col in range(N_COL):
                    osc_array[row][col].set_init_val(node_init[row, col], 0)

            x.append(mapping_fn(graph))
            y.append(ideal_pattern)

        x, y = jnp.array(x), jnp.array(y)

        noise_seed = jnp.array(np.random.randint(0, 2**32 - 1, size=batch_size))
        # print(f"loss: {periodic_mse(x, y):.4f}")
        yield x, noise_seed, y


def raw_to_rel_phase(raw_phase_out: jax.Array):
    """Convert the raw phase to the relative phase"""
    n_repeat = N_NODE - 1
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
    return jnp.mean(jnp.max(jnp.square(phase_diff - y), axis=1))
    # return jnp.mean(jnp.square(phase_diff - y))


def min_rand_reconstruction_loss(
    model: BaseAnalogCkt, x: jax.Array, noise_seed: jax.Array, diff_fn: Callable
):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0))(
        time_info, x, [], 0, noise_seed
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
    diff_fn: Callable,
):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0))(
        time_info, x, [], 0, noise_seed
    )
    return diff_fn(y_raw[:, -1, :], y)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    loss_fn: Callable,
    data: list[jax.Array],
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, *data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def plot_evolution(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str = None,
    plot_time: list[int] = PLOT_TIME,
):
    """Plot the evolution of the oscillator phase"""

    x_init, noise_seed = data[0], data[1]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0))(
        time_info, x_init, [], 0, noise_seed
    )

    for i in range(y_raw.shape[0]):
        # phase is periodic over 2
        y_readout = y_raw[i] % 2

        fig, ax = plt.subplots(ncols=len(plot_time))
        for j, time in enumerate(plot_time):
            y_readout_t = y_readout[time].reshape(N_ROW, N_COL)
            y_readout_t = jnp.abs(y_readout_t - y_readout_t[0, 0])
            # Calculate the phase difference
            phase_diff = jnp.where(y_readout_t > 1, 2 - y_readout_t, y_readout_t)
            ax[j].imshow(phase_diff, cmap="gray", vmin=0, vmax=1)

        di = [d[i : i + 1] for d in data]
        loss = loss_fn(model, *di)
        plt.suptitle(title + f", Loss: {loss:.4f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    trainable_mgr = TrainableMananger()

    graph = CDG()
    nodes = [[None for _ in range(N_COL)] for _ in range(N_ROW)]
    row_edges = [[None for _ in range(N_COL - 1)] for _ in range(N_ROW)]
    col_edges = [[None for _ in range(N_COL)] for _ in range(N_ROW - 1)]

    # Initialze nodes
    for row in range(N_ROW):
        for col in range(N_COL):
            node = Osc_modified(
                lock_strength=1.2,
                cpl_strength=2,
                lock_fn=locking_fn,
                osc_fn=coupling_fn,
            )
            self_edge = Coupling(k=1)
            graph.connect(self_edge, node, node)
            nodes[row][col] = node

    # Connect neighboring nodes
    for row in range(N_ROW):
        for col in range(N_COL - 1):
            edge = Coupling(k=trainable_mgr.new_var())
            graph.connect(edge, nodes[row][col], nodes[row][col + 1])
            row_edges[row][col] = edge

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            edge = Coupling(k=trainable_mgr.new_var())
            graph.connect(edge, nodes[row][col], nodes[row + 1][col])
            col_edges[row][col] = edge

    # flatten the nodes for readout
    nodes_flat = [node for row in nodes for node in row]

    rec_circuit_class = OptCompiler().compile(
        "rec",
        graph,
        obc_spec,
        readout_nodes=nodes_flat,
        normalize_weight=False,
        do_clipping=False,
    )

    if DIFF_FN == "periodic_mse":
        diff_fn = periodic_mse
    elif DIFF_FN == "normalize_angular_diff":
        diff_fn = normalize_angular_diff

    if TASK == "one-to-one":
        dl = partial(
            dataloader2,
            graph=graph,
            osc_array=nodes,
            mapping_fn=rec_circuit_class.cdg_to_initial_states,
            snp_prob=SNP_PROB,
            gauss_std=GAUSS_STD,
        )
        loss_fn = partial(pattern_reconstruction_loss, diff_fn=diff_fn)
    elif TASK == "rand-to-many":
        dl, loss_fn = dataloader, partial(min_rand_reconstruction_loss, diff_fn=diff_fn)

    for seed in range(10):
        np.random.seed(seed)
        train_loss_best = 1e9

        edge_init = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            edge_init += pattern_to_edge_initialization(NUMBERS[i])

        edge_init /= N_CLASS

        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=jnp.array(edge_init),
            # init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
            is_stochastic=False,
            solver=Heun(),
        )

        #
        plot_data = next(dl(PLOT_BZ))
        plot_evolution(
            model=model,
            loss_fn=loss_fn,
            data=plot_data,
            title="Before training",
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        print(f"\n\nSeed {seed}")
        # print(f"Weights: {model.trainable}")

        for step, data in zip(range(STEPS), dl(BZ)):

            model, opt_state, train_loss = make_step(model, opt_state, loss_fn, data)
            # if step == 0:
            # print(f"Initial Loss: {train_loss}")
            print(f"\tStep {step}, Loss: {train_loss}")
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                best_weight = model.trainable.copy()

        # print(f"\tBest Loss: {train_loss_best}")
        print(f"Best Weights: {best_weight}")

        plot_evolution(
            model=model,
            loss_fn=loss_fn,
            data=plot_data,
            title="After training",
        )

        # Test the best model w/o noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=False,
            solver=Heun(),
        )

        for _, data in zip(range(1), dl(VALIDATION_BZ)):
            val_loss = loss_fn(model, *data)
            print(f"Validation Loss w/o noise: {val_loss}")

        # Test the best model w/ noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=True,
            solver=Heun(),
        )

        plot_evolution(
            model=model,
            loss_fn=loss_fn,
            data=plot_data,
            title="Noisy (before fine-tune)",
        )

        for _, data in zip(range(1), dl(VALIDATION_BZ)):
            val_loss = loss_fn(model, *data)
            print(f"Validation Loss w/ noise: {val_loss}")

        # Double the weight for the noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight * 2,
            is_stochastic=True,
            solver=Heun(),
        )

        for _, data in zip(range(1), dl(VALIDATION_BZ)):
            val_loss = loss_fn(model, *data)
            print(f"Validation Loss w/ noise (double weight): {val_loss}")

        # Fine-tune the best model w/ noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=True,
            solver=Heun(),
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        train_loss_best = 1e9

        for step, data in zip(range(STEPS), dl(BZ)):
            model, opt_state, train_loss = make_step(model, opt_state, loss_fn, data)
            print(f"\tFine-tune Step {step}, Loss: {train_loss}")
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                best_weight = model.trainable.copy()

        # print(f"\tFine-tune Best Loss: {train_loss_best}")
        print(f"Fine-tune Best Weights: {best_weight}")

        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=True,
            solver=Heun(),
        )

        plot_evolution(
            model=model,
            loss_fn=loss_fn,
            data=plot_data,
            title="Noisy (after fine-tune)",
        )

        # Test the best model w/ noise
        for _, data in zip(range(1), dl(VALIDATION_BZ)):
            val_loss = loss_fn(model, *data)
            print(f"Validation Loss w/ noise (fine-tune): {val_loss}")
