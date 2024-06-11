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
from pattern_recog_dataloader import NUMBERS, dataloader, dataloader2
from pattern_recog_loss import (
    min_rand_reconstruction_loss,
    normalize_angular_diff,
    pattern_reconstruction_loss,
    periodic_mean_max_se,
    periodic_mse,
)
from spec_optimization import (
    Coupling,
    Osc_modified,
    T,
    coupling_fn,
    locking_fn,
    obc_spec,
)

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import TrainableMgr

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
            best_weight = model.a_trainable.copy()

        if USE_WANDB:
            wandb.log(
                data={
                    f"{log_prefix}_train_loss": train_loss,
                    f"{log_prefix}_val_loss": val_loss,
                },
            )

    return val_loss_best, best_weight


if __name__ == "__main__":

    trainable_mgr = TrainableMgr()

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
            edge = Coupling(k=trainable_mgr.new_analog())
            graph.connect(edge, nodes[row][col], nodes[row][col + 1])
            row_edges[row][col] = edge

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            edge = Coupling(k=trainable_mgr.new_analog())
            graph.connect(edge, nodes[row][col], nodes[row + 1][col])
            col_edges[row][col] = edge

    # flatten the nodes for readout
    nodes_flat = [node for row in nodes for node in row]

    rec_circuit_class = OptCompiler().compile(
        "rec",
        graph,
        obc_spec,
        trainable_mgr=trainable_mgr,
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
            n_class=N_CLASS,
            graph=graph,
            osc_array=nodes,
            mapping_fn=rec_circuit_class.cdg_to_initial_states,
            snp_prob=SNP_PROB,
            gauss_std=GAUSS_STD,
        )
        loss_fn = partial(
            pattern_reconstruction_loss, time_info=time_info, diff_fn=diff_fn
        )
    elif TASK == "rand-to-many":
        dl = partial(dataloader, n_node=N_NODE)
        loss_fn = partial(
            min_rand_reconstruction_loss,
            time_info=time_info,
            diff_fn=diff_fn,
            N_CLASS=N_CLASS,
        )

    np.random.seed(SEED)

    if WEIGHT_INIT == "hebbian":
        edge_init = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            edge_init += pattern_to_edge_initialization(NUMBERS[i])

        edge_init /= N_CLASS
    elif WEIGHT_INIT == "random":
        edge_init = np.random.normal(size=len(trainable_mgr.analog))

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
        best_weight = model.a_trainable

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
