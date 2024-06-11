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
from pattern_recog_parser import args
from spec_optimization import (
    Coupling,
    Cpl_digital,
    Osc_modified,
    T,
    coupling_fn,
    locking_fn,
    obc_spec,
)

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr

jax.config.update("jax_enable_x64", True)

N_ROW, N_COL = 5, 3
N_NODE = N_ROW * N_COL
N_EDGE = (N_ROW - 1) * N_COL + (N_COL - 1) * N_ROW

N_CLASS = args.n_class
N_CYCLES = args.n_cycle

SEED = args.seed

WEIGHT_INIT = args.weight_init
WEIGHT_BITS = args.weight_bits

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

USE_HARD_GUMBEL = args.hard_gumbel
GUMBEL_TEMP_START, GUMBEL_TEMP_END = args.gumbel_temp_start, args.gumbel_temp_end

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

    row_init = [[None for _ in range(N_COL - 1)] for _ in range(N_ROW)]
    col_init = [[None for _ in range(N_COL)] for _ in range(N_ROW - 1)]

    for row in range(N_ROW):
        for col in range(N_COL - 1):
            diff = pattern[row, col] - pattern[row, col + 1]
            row_init[row][col] = -1.0 if diff else 1.0

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            diff = pattern[row, col] - pattern[row + 1, col]
            col_init[row][col] = -1.0 if diff else 1.0

    return np.array(row_init), np.array(col_init)


def edge_init_to_trainable_init(
    row_init: np.ndarray,
    col_init: np.ndarray,
    k_rows: list[list[Trainable]],
    k_cols: list[list[Trainable]],
    trainable_mgr: TrainableMgr,
) -> tuple[jax.Array, list[jax.Array]]:
    """Convert the edge initialization to trainable initialization"""

    def one_hot_digitize(x: np.ndarray, bins: np.ndarray):
        # Digitize with the nearest bin and then one-hot encode
        # the index is then one-hot encoded
        nearest_bins = np.abs(x[:, :, None] - bins[None, None, :]).argmin(axis=-1)
        one_hot = jax.nn.one_hot(nearest_bins, bins.shape[0])
        return one_hot

    row_init_quantized, col_init_quantized = row_init, col_init

    if WEIGHT_BITS is not None:
        weight_choices: list = Cpl_digital.attr_def["k"].attr_type.val_choices

        # normalize the weight choices to between -1 and 1
        normalize_factor = 2 ** (WEIGHT_BITS - 1)
        weight_choices = np.array(weight_choices) / normalize_factor

        # If digital setup, the only analog trainable is the lock strength
        trainable_mgr.analog[0].init_val = 1.2 / normalize_factor

        row_init_quantized = one_hot_digitize(row_init, weight_choices)
        col_init_quantized = one_hot_digitize(col_init, weight_choices)

    for row in range(N_ROW):
        for col in range(N_COL - 1):
            k_rows[row][col].init_val = row_init_quantized[row, col]

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            k_cols[row][col].init_val = col_init_quantized[row, col]

    a_trainable = trainable_mgr.get_initial_vals("analog")
    d_trainable = trainable_mgr.get_initial_vals("digital")

    return (a_trainable, d_trainable)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    loss_fn: Callable,
    data: list[jax.Array],
    gumbel_temp: float,
):
    train_data, val_data = [d[:TRAIN_BZ] for d in data], [d[TRAIN_BZ:] for d in data]
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, *train_data, gumbel_temp
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    val_loss = loss_fn(model, *val_data, gumbel_temp)
    return model, opt_state, train_loss, val_loss


def plot_evolution(
    model_cls: type,
    best_weight: tuple[jax.Array, list[jax.Array]],
    is_stochastic: bool,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str = None,
    gumbel_temp: float = 1,
):
    """Plot the evolution of the oscillator phase"""
    if PLOT_EVOLVE == 0:
        return

    model: BaseAnalogCkt = model_cls(
        init_trainable=best_weight,
        is_stochastic=is_stochastic,
        solver=Heun(),
        hard_gumbel=True,
    )

    x_init, noise_seed = data[0], data[1]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0, None))(
        time_info, x_init, [], 0, noise_seed, gumbel_temp
    )
    plot_time = [i for i in range(len(saveat))]

    n_row, n_col = y_raw.shape[0], len(plot_time)
    fig, ax = plt.subplots(
        ncols=len(plot_time), nrows=y_raw.shape[0], figsize=(n_col, n_row * 1.75)
    )
    losses = []
    for i, y in enumerate(y_raw):
        # phase is periodic over 2
        y_readout = y % 2

        for j, time in enumerate(plot_time):
            y_readout_t = y_readout[time].reshape(N_ROW, N_COL)
            y_readout_t = jnp.abs(y_readout_t - y_readout_t[0, 0])
            # Calculate the phase difference
            phase_diff = jnp.where(y_readout_t > 1, 2 - y_readout_t, y_readout_t)
            ax[i, j].axis("off")
            ax[i, j].imshow(phase_diff, cmap="gray", vmin=0, vmax=1)

        di = [d[i : i + 1] for d in data]
        losses.append(loss_fn(model, *di, gumbel_temp))
    loss = jnp.mean(jnp.array(losses))
    plt.suptitle(title + f", Loss: {loss:.4f}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if USE_WANDB:
        wandb.log(data={f"{title}_evolution": plt}, commit=False)
        plt.close()
    else:
        plt.show()


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator, log_prefix: str = ""):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    val_loss_best = 1e9

    gumbel_temp = GUMBEL_TEMP_START
    gumbel_temp_decay = (GUMBEL_TEMP_START - GUMBEL_TEMP_END) / STEPS

    for step, data in zip(range(STEPS), dl(BZ)):

        if step == 0:  # Set up the baseline loss
            train_data = [d[:TRAIN_BZ] for d in data]
            val_data = [d[TRAIN_BZ:] for d in data]
            train_loss = loss_fn(model, *train_data, GUMBEL_TEMP_END)
            val_loss = loss_fn(model, *val_data, GUMBEL_TEMP_END)

        else:
            model, opt_state, train_loss, val_loss = make_step(
                model, opt_state, loss_fn, data, gumbel_temp
            )

        print(f"Step {step}, Train loss: {train_loss}, Val loss: {val_loss}")
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            best_weight = (model.a_trainable.copy(), model.d_trainable.copy())

        if USE_WANDB:
            wandb.log(
                data={
                    f"{log_prefix}_train_loss": train_loss,
                    f"{log_prefix}_val_loss": val_loss,
                    "gumbel_temp": gumbel_temp,
                },
            )
        gumbel_temp -= gumbel_temp_decay

    return val_loss_best, best_weight


if __name__ == "__main__":

    trainable_mgr = TrainableMgr()

    graph = CDG()
    nodes = [[None for _ in range(N_COL)] for _ in range(N_ROW)]
    row_edges = [[None for _ in range(N_COL - 1)] for _ in range(N_ROW)]
    col_edges = [[None for _ in range(N_COL)] for _ in range(N_ROW - 1)]

    if not WEIGHT_BITS:
        lock_val = 1.2
        cpl_type = Coupling
    else:
        lock_val = trainable_mgr.new_analog()
        cpl_type = Cpl_digital

    # Store all the trainable parameters
    row_ks = [
        [
            (
                trainable_mgr.new_analog()
                if not WEIGHT_BITS
                else trainable_mgr.new_digital()
            )
            for _ in range(N_COL - 1)
        ]
        for _ in range(N_ROW)
    ]
    col_ks = [
        [
            (
                trainable_mgr.new_analog()
                if not WEIGHT_BITS
                else trainable_mgr.new_digital()
            )
            for _ in range(N_COL)
        ]
        for _ in range(N_ROW - 1)
    ]

    # Initialze nodes
    for row in range(N_ROW):
        for col in range(N_COL):
            node = Osc_modified(
                lock_strength=lock_val,
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
            edge = cpl_type(k=row_ks[row][col])
            graph.connect(edge, nodes[row][col], nodes[row][col + 1])
            row_edges[row][col] = edge

    for row in range(N_ROW - 1):
        for col in range(N_COL):
            edge = cpl_type(k=col_ks[row][col])
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
        hard_gumbel=USE_HARD_GUMBEL,
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
        row_init, col_init = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            r, c = pattern_to_edge_initialization(NUMBERS[i])
            row_init += r
            col_init += c

        row_init /= N_CLASS
        col_init /= N_CLASS
    elif WEIGHT_INIT == "random":
        row_init = np.random.normal(size=(N_ROW, N_COL - 1))
        col_init = np.random.normal(size=(N_ROW - 1, N_COL))

    trainable_init = edge_init_to_trainable_init(
        row_init, col_init, row_ks, col_ks, trainable_mgr
    )

    plot_data = next(dl(PLOT_BZ))

    plot_evolution(
        model_cls=rec_circuit_class,
        best_weight=trainable_init,
        is_stochastic=False,
        loss_fn=loss_fn,
        data=plot_data,
        title="Before training",
    )

    if not NO_NOISELESS_TRAIN:

        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=trainable_init,
            is_stochastic=False,
            solver=Heun(),
            hard_gumbel=USE_HARD_GUMBEL,
        )
        best_loss, best_weight = train(model, loss_fn, dl, "tran_noiseless")

        print(f"Best Loss: {best_loss}")
        print(f"Best Weights: {best_weight}")

        plot_evolution(
            model_cls=rec_circuit_class,
            best_weight=best_weight,
            is_stochastic=False,
            loss_fn=loss_fn,
            data=plot_data,
            title="After training",
        )
    else:
        best_weight = trainable_init

    plot_evolution(
        model_cls=rec_circuit_class,
        best_weight=best_weight,
        is_stochastic=True,
        loss_fn=loss_fn,
        data=plot_data,
        title="Noisy (before fine-tune)",
    )

    # Fine-tune the best model w/ noise
    model: BaseAnalogCkt = rec_circuit_class(
        init_trainable=best_weight,
        is_stochastic=True,
        solver=Heun(),
        hard_gumbel=USE_HARD_GUMBEL,
    )

    best_loss, best_weight = train(model, loss_fn, dl, "tran_noisy")
    print(f"\tFine-tune Best Loss: {best_loss}")
    print(f"Fine-tune Best Weights: {best_weight}")

    # Model after fine-tune
    plot_evolution(
        model_cls=rec_circuit_class,
        best_weight=best_weight,
        is_stochastic=True,
        loss_fn=loss_fn,
        data=plot_data,
        title="Noisy (after fine-tune)",
    )
    wandb.finish()
