from argparse import ArgumentParser
from functools import partial
from typing import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from diffrax import Heun
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
parser = ArgumentParser()

# Example command: python pattern_recog_digit.py --gauss_std 0.1 --trans_noise_std 0.1
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default="one-to-one")
parser.add_argument(
    "--n_cycle",
    type=int,
    default=1,
    help="Number of cycles to wait for the oscillators to read out",
)
parser.add_argument(
    "--weight_init",
    type=str,
    default="hebbian",
    choices=["hebbian", "random"],
    help="Method to initialize training weights.",
)
parser.add_argument(
    "--diff_fn",
    type=str,
    choices=["periodic_mse", "periodic_mean_max_se", "normalize_angular_diff"],
    default="periodic_mean_max_se",
    help="The function to evaluate the difference between the readout and the target",
)
parser.add_argument(
    "--point_per_cycle", type=int, default=50, help="Number of time points per cycle"
)
parser.add_argument("--snp_prob", type=float, default=0.0, help="Salt-and-pepper noise")
parser.add_argument("--gauss_std", type=float, default=0.0, help="Gaussian noise std")
parser.add_argument(
    "--trans_noise_std", type=float, default=0.0, help="Transition noise std"
)
parser.add_argument(
    "--n_class", type=int, default=5, help="Number of classes to recognize"
)
parser.add_argument("--steps", type=int, default=32, help="Number of training steps")
parser.add_argument("--bz", type=int, default=512, help="Batch size")
parser.add_argument(
    "--validation_split", type=float, default=0.5, help="Validation split ratio"
)
parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
parser.add_argument(
    "--optimizer", type=str, default="adam", help="Type of the optimizer"
)
parser.add_argument(
    "--plot_evolve",
    type=int,
    default=0,
    help="Number of time points to plot the evolution",
)
parser.add_argument(
    "--no_noiseless_train", action="store_true", help="Skip noiseless training"
)
parser.add_argument("--wandb", action="store_true", help="Log to wandb")

args = parser.parse_args()


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

N_CLASS = args.n_class
N_CYCLES = args.n_cycle

SEED = args.seed

WEIGHT_INIT = args.weight_init
POINT_PER_CYCLE = args.point_per_cycle
PLOT_EVOLVE = args.plot_evolve

LEARNING_RATE = args.lr
STEPS = args.steps
BZ = args.bz
VALIDATION_SPLIT = args.validation_split
VALIDATION_BZ = int(BZ * VALIDATION_SPLIT)
TRAIN_BZ = BZ - VALIDATION_BZ

OPTIMIZER = args.optimizer
optim = getattr(optax, OPTIMIZER)(LEARNING_RATE)

SNP_PROB, GAUSS_STD = args.snp_prob, args.gauss_std
TRANS_NOISE_STD = args.trans_noise_std

USE_WANDB = args.wandb
TASK = args.task
DIFF_FN = args.diff_fn

NO_NOISELESS_TRAIN = args.no_noiseless_train


if PLOT_EVOLVE != 0:
    saveat = jnp.linspace(0, T * N_CYCLES, PLOT_EVOLVE, endpoint=True)
else:
    saveat = [T * N_CYCLES]

time_info = TimeInfo(
    t0=0,
    t1=T * N_CYCLES,
    dt0=T / POINT_PER_CYCLE,
    saveat=saveat,
)

# Hard coded the noise std to the production rules of
# Osc_modified - Cpl - Osc_modified
obc_spec.production_rules()[3]._noise_exp = TRANS_NOISE_STD
obc_spec.production_rules()[4]._noise_exp = TRANS_NOISE_STD

PLOT_BZ = 4

if USE_WANDB:
    wandb_run = wandb.init(project="obc", config=vars(args), tags=["digit_recognition"])


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


@eqx.filter_jit
def raw_to_rel_phase(raw_phase_out: jax.Array):
    """Convert the raw phase to the relative phase"""
    n_repeat = N_NODE - 1
    rel_phase = raw_phase_out[:, 1:] - raw_phase_out[:, 0].repeat(n_repeat).reshape(
        -1, n_repeat
    )
    return rel_phase


@eqx.filter_jit
def normalize_angular_diff(y_end_readout: jax.Array, y: jax.Array):
    x = raw_to_rel_phase(y_end_readout)
    return jnp.sin(jnp.pi * ((x - y) / 2 % 1))


@eqx.filter_jit
def periodic_mse(y_end_readout: jax.Array, y: jax.Array):
    y_end_readout = y_end_readout % 2
    rel_y = jnp.abs(raw_to_rel_phase(y_end_readout))
    phase_diff = jnp.where(rel_y > 1, 2 - rel_y, rel_y)
    return jnp.mean(jnp.square(phase_diff - y))


@eqx.filter_jit
def periodic_mean_max_se(y_end_readout: jax.Array, y: jax.Array):
    y_end_readout = y_end_readout % 2
    rel_y = jnp.abs(raw_to_rel_phase(y_end_readout))
    phase_diff = jnp.where(rel_y > 1, 2 - rel_y, rel_y)
    return jnp.mean(jnp.max(jnp.square(phase_diff - y), axis=1))


@eqx.filter_jit
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


@eqx.filter_jit
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
    train_data, val_data = [d[:TRAIN_BZ] for d in data], [d[TRAIN_BZ:] for d in data]
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *train_data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    val_loss = loss_fn(model, *val_data)
    return model, opt_state, train_loss, val_loss


def plot_evolution(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str = None,
):
    """Plot the evolution of the oscillator phase"""
    if PLOT_EVOLVE == 0:
        return

    x_init, noise_seed = data[0], data[1]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0))(
        time_info, x_init, [], 0, noise_seed
    )
    plot_time = [i for i in range(len(saveat))]

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


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator, log_prefix: str = ""):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    val_loss_best = 1e9

    for step, data in zip(range(STEPS), dl(BZ)):

        if step == 0:  # Set up the baseline loss
            train_data = [d[:TRAIN_BZ] for d in data]
            val_data = [d[TRAIN_BZ:] for d in data]
            train_loss = loss_fn(model, *train_data)
            val_loss = loss_fn(model, *val_data)

        else:
            model, opt_state, train_loss, val_loss = make_step(
                model, opt_state, loss_fn, data
            )

        print(f"Step {step}, Train loss: {train_loss}, Val loss: {val_loss}")
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            best_weight = model.trainable.copy()

        if USE_WANDB:
            wandb.log(
                data={
                    f"{log_prefix}_train_loss": train_loss,
                    f"{log_prefix}_val_loss": val_loss,
                },
            )

    return val_loss_best, best_weight


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
    elif DIFF_FN == "periodic_mean_max_se":
        diff_fn = periodic_mean_max_se
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

    np.random.seed(SEED)

    if WEIGHT_INIT == "hebbian":
        edge_init = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            edge_init += pattern_to_edge_initialization(NUMBERS[i])

        edge_init /= N_CLASS
    elif WEIGHT_INIT == "random":
        edge_init = np.random.normal(size=trainable_mgr.idx + 1)

    model: BaseAnalogCkt = rec_circuit_class(
        init_trainable=jnp.array(edge_init),
        # init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
        is_stochastic=False,
        solver=Heun(),
    )

    plot_data = next(dl(PLOT_BZ))
    plot_evolution(
        model=model,
        loss_fn=loss_fn,
        data=plot_data,
        title="Before training",
    )

    if not NO_NOISELESS_TRAIN:
        best_loss, best_weight = train(model, loss_fn, dl, "tran_noiseless")

        print(f"Best Loss: {best_loss}")
        print(f"Best Weights: {best_weight}")

        plot_evolution(
            model=model,
            loss_fn=loss_fn,
            data=plot_data,
            title="After training",
        )
    else:
        best_weight = model.trainable

    # Fine-tune the best model w/ noise
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

    best_loss, best_weight = train(model, loss_fn, dl, "tran_noisy")
    print(f"\tFine-tune Best Loss: {best_loss}")
    print(f"Fine-tune Best Weights: {best_weight}")

    plot_evolution(
        model=model,
        loss_fn=loss_fn,
        data=plot_data,
        title="Noisy (after fine-tune)",
    )
