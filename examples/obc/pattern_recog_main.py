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
from pattern_recog_dataloader import NUMBERS_5x3, NUMBERS_10x6, dataloader, dataloader2
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

PATTERN_SHAPE = args.pattern_shape

if PATTERN_SHAPE == "5x3":
    N_ROW, N_COL = 5, 3
    NUMBERS = NUMBERS_5x3
elif PATTERN_SHAPE == "10x6":
    N_ROW, N_COL = 10, 6
    NUMBERS = NUMBERS_10x6

N_NODE = N_ROW * N_COL
N_EDGE = (N_ROW - 1) * N_COL + (N_COL - 1) * N_ROW

N_CLASS = args.n_class
N_CYCLES = args.n_cycle

SEED = args.seed
np.random.seed(SEED)

WEIGHT_INIT = args.weight_init
WEIGHT_BITS = args.weight_bits

POINT_PER_CYCLE = args.point_per_cycle
PLOT_EVOLVE = args.plot_evolve
PLOT_BZ = args.num_plot

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
GUMBEL_SHEDULE = args.gumbel_schedule

TRAINABLE_LOCKING = args.trainable_locking
INIT_LOCK_STRENGTH = args.locking_strength
TRAINABLE_COUPLING = args.trainable_coupling
INIT_COUPLING_STRENGTH = args.coupling_strength

trainable_mgr = TrainableMgr()
if TRAINABLE_LOCKING:
    lock_var = trainable_mgr.new_analog()
    lock_var.init_val = INIT_LOCK_STRENGTH
else:
    lock_var = INIT_LOCK_STRENGTH

if TRAINABLE_COUPLING:
    cpl_var = trainable_mgr.new_analog()
    cpl_var.init_val = INIT_COUPLING_STRENGTH
else:
    cpl_var = INIT_COUPLING_STRENGTH


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

if USE_WANDB:
    wandb_run = wandb.init(project="obc", config=vars(args), tags=["digit_recognition"])


def node_idx_to_coord(idx: int):
    return idx // N_COL, idx % N_COL


def coord_to_node_idx(row: int, col: int):
    return row * N_COL + col


def pattern_to_edge_initialization(pattern: np.ndarray) -> np.ndarray:
    """Map the pattern to the edge initialization

    If two pixels are different, the edge is initialized with -1.0
    Otherwise, the edge is initialized with 1.0
    """

    init_weights = np.zeros((N_NODE, N_NODE))
    for node_idx in range(N_NODE):
        for n_node_idx in range(N_NODE):

            row, col = node_idx_to_coord(node_idx)
            n_row, n_col = node_idx_to_coord(n_node_idx)
            diff = pattern[row, col] - pattern[n_row, n_col]
            init_weights[node_idx, n_node_idx] = -1.0 if diff else 1.0

    return init_weights


def edge_init_to_trainable_init(
    init_weights: np.ndarray,
    edge_ks: list[list[Trainable]],
    trainable_mgr: TrainableMgr,
) -> tuple[jax.Array, list[jax.Array]]:
    """Convert the edge initialization to trainable initialization

    Enumerate through the edge_ks; If not None, set the initial value
    according to the init_weights
    """

    def one_hot_digitize(x: np.ndarray, bins: np.ndarray):
        # Digitize with the nearest bin and then one-hot encode
        # the index is then one-hot encoded
        nearest_bins = np.abs(x[:, :, None] - bins[None, None, :]).argmin(axis=-1)
        one_hot = jax.nn.one_hot(nearest_bins, bins.shape[0])
        return one_hot

    init_weights_quantized = init_weights

    if WEIGHT_BITS is not None:
        weight_choices: list = Cpl_digital.attr_def["k"].attr_type.val_choices

        # normalize the weight choices to between -1 and 1
        weight_choices = np.array(weight_choices)

        init_weights_quantized = one_hot_digitize(init_weights, weight_choices)

    for node_idx, node_kss in enumerate(edge_ks):
        for next_node_idx, k in enumerate(node_kss):
            if k is None or next_node_idx <= node_idx:
                continue
            k.init_val = init_weights_quantized[node_idx, next_node_idx]

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
    hard_gumbel: bool = False,
):
    train_data, val_data = [d[:TRAIN_BZ] for d in data], [d[TRAIN_BZ:] for d in data]
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, *train_data, gumbel_temp, hard_gumbel
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    # When validating, always use hard gumbel to force the param to be physical
    val_loss = loss_fn(model, *val_data, gumbel_temp, hard_gumbel=True)
    return model, opt_state, train_loss, val_loss


def plot_evolution(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str,
    gumbel_temp: float,
    hard_gumbel: bool = True,
):
    x_init, args_seed, noise_seed = data[0], data[1], data[2]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0, None, None))(
        time_info, x_init, [], args_seed, noise_seed, gumbel_temp, hard_gumbel
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
            ax[i, j].imshow(phase_diff, cmap="gray_r", vmin=0, vmax=1)

        di = [d[i : i + 1] for d in data]
        losses.append(loss_fn(model, *di, gumbel_temp, hard_gumbel))
    loss = jnp.mean(jnp.array(losses))
    plt.suptitle(title + f", Loss: {loss:.4f}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if USE_WANDB:
        wandb.log(data={f"{title}_evolution": plt}, commit=False)
        plt.close()
    else:
        plt.show()


def load_model_and_plot(
    model_cls: type,
    best_weight: tuple[jax.Array, list[jax.Array]],
    is_stochastic: bool,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str = None,
    gumbel_temp: float = None,
):
    """Plot the evolution of the oscillator phase"""
    if PLOT_EVOLVE == 0:
        return

    hard_gumbel = True
    if gumbel_temp is not None:
        hard_gumbel = False
    else:
        gumbel_temp = 1  # Set to a value to avoid divide by None
    model: BaseAnalogCkt = model_cls(
        init_trainable=best_weight,
        is_stochastic=is_stochastic,
        solver=Heun(),
    )

    plot_evolution(model, loss_fn, data, title, gumbel_temp, hard_gumbel)


def linear_schedule(step: int, start: float, end: float, tot_steps: int):
    return start - (start - end) * step / (tot_steps - 1)


def exp_schedule(step: int, start: float, end: float, tot_steps: int):
    return np.exp(-step / (tot_steps - 1) * np.log(start / end)) * start


if GUMBEL_SHEDULE == "linear":
    next_gumbel_temp = partial(
        linear_schedule, start=GUMBEL_TEMP_START, end=GUMBEL_TEMP_END, tot_steps=STEPS
    )
elif GUMBEL_SHEDULE == "exp":
    next_gumbel_temp = partial(
        exp_schedule, start=GUMBEL_TEMP_START, end=GUMBEL_TEMP_END, tot_steps=STEPS
    )


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator, log_prefix: str = ""):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    val_loss_best = 1e9

    gumbel_temp = GUMBEL_TEMP_START
    hard_gumbel = USE_HARD_GUMBEL

    for step, data in zip(range(STEPS), dl(BZ)):

        if step == 0:  # Set up the baseline loss
            train_data = [d[:TRAIN_BZ] for d in data]
            val_data = [d[TRAIN_BZ:] for d in data]
            train_loss = loss_fn(model, *train_data, gumbel_temp, hard_gumbel=True)
            val_loss = loss_fn(model, *val_data, gumbel_temp, hard_gumbel=True)

        else:
            model, opt_state, train_loss, val_loss = make_step(
                model, opt_state, loss_fn, data, gumbel_temp, hard_gumbel
            )

        print(
            f"Step {step}, Train loss: {train_loss}, Val loss: {val_loss}, Gumbel temp: {gumbel_temp}"
        )
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
        gumbel_temp = next_gumbel_temp(step + 1)

    return val_loss_best, best_weight


if __name__ == "__main__":

    graph = CDG()
    nodes = [[None for _ in range(N_COL)] for _ in range(N_ROW)]

    if not WEIGHT_BITS:
        cpl_type = Coupling
    else:
        cpl_type = Cpl_digital

    # Store all the trainable parameters with connectivity matrix
    # Connect 8 neighbor nodes
    edge_ks = [[None for _ in range(N_NODE)] for _ in range(N_NODE)]
    for row in range(N_ROW):
        for col in range(N_COL):
            # Connect to right, bottom, bottom-right, bottom-left
            node_idx = coord_to_node_idx(row, col)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                next_row, next_col = row + dr, col + dc
                if (
                    next_row < 0
                    or next_row >= N_ROW
                    or next_col < 0
                    or next_col >= N_COL
                ):
                    continue

                new_var = (
                    trainable_mgr.new_analog()
                    if not WEIGHT_BITS
                    else trainable_mgr.new_digital()
                )
                next_node_idx = coord_to_node_idx(next_row, next_col)
                edge_ks[node_idx][next_node_idx] = edge_ks[next_node_idx][node_idx] = (
                    new_var
                )

    # Initialze nodes
    for row in range(N_ROW):
        for col in range(N_COL):
            node = Osc_modified(
                lock_strength=lock_var,
                cpl_strength=cpl_var,
                lock_fn=locking_fn,
                osc_fn=coupling_fn,
            )
            self_edge = Coupling(k=1.0)
            graph.connect(self_edge, node, node)
            nodes[row][col] = node

    # Connect neighboring nodes
    for node_idx, node_kss in enumerate(edge_ks):
        row, col = node_idx_to_coord(node_idx)
        for next_node_idx, k in enumerate(node_kss):
            if k is None or next_node_idx <= node_idx:
                continue
            next_row, next_col = node_idx_to_coord(next_node_idx)
            edge = cpl_type(k=k)
            graph.connect(edge, nodes[row][col], nodes[next_row][next_col])

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

    if WEIGHT_INIT == "hebbian":
        init_weights = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            w = pattern_to_edge_initialization(NUMBERS[i])
            init_weights += w

        init_weights /= N_CLASS
    elif WEIGHT_INIT == "random":
        init_weights = np.random.normal(size=(N_NODE, N_NODE))

    trainable_init = edge_init_to_trainable_init(init_weights, edge_ks, trainable_mgr)

    plot_data = next(dl(PLOT_BZ))

    load_model_and_plot(
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
        )
        best_loss, best_weight = train(model, loss_fn, dl, "tran_noiseless")

        print(f"Best Loss: {best_loss}")
        print(f"Best Weights: {best_weight}")

        load_model_and_plot(
            model_cls=rec_circuit_class,
            best_weight=best_weight,
            is_stochastic=False,
            loss_fn=loss_fn,
            data=plot_data,
            title="After training",
        )
    else:
        best_weight = trainable_init

    load_model_and_plot(
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
    )

    best_loss, best_weight = train(model, loss_fn, dl, "tran_noisy")
    print(f"\tFine-tune Best Loss: {best_loss}")
    print(f"Fine-tune Best Weights: {best_weight}")

    # Model after fine-tune
    load_model_and_plot(
        model_cls=rec_circuit_class,
        best_weight=best_weight,
        is_stochastic=True,
        loss_fn=loss_fn,
        data=plot_data,
        title="Noisy (after fine-tune)",
    )
    wandb.finish()
