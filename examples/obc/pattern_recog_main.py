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
from pattern_recog_matrix_solve import OscillatorNetworkMatrixSolve
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

MATRIX_SOLVE = args.matrix_solve
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

CONNECTION = args.connection
TRAINALBE_CONNECTION = args.trainable_connection

SEED = args.seed
TEST_SEED = args.test_seed
TEST_BZ = args.test_bz
assert (
    SEED != TEST_SEED
), f"Training seed {SEED} is the same as the testing seed {TEST_SEED}"
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
UNIFORM_NOISE = args.uniform_noise

USE_WANDB = args.wandb
TAG = args.tag
TASK = args.task
DIFF_FN = args.diff_fn
L1_NORM_WEIGHT = args.l1_norm_weight

NO_NOISELESS_TRAIN = args.no_noiseless_train

USE_HARD_GUMBEL = args.hard_gumbel
GUMBEL_TEMP_START, GUMBEL_TEMP_END = args.gumbel_temp_start, args.gumbel_temp_end
GUMBEL_SHEDULE = args.gumbel_schedule

FIX_COUPLING_WEIGHT = args.fix_coupling_weight
if FIX_COUPLING_WEIGHT and WEIGHT_BITS:
    raise NotImplementedError(
        "Digital weight is not supported for fixed coupling weight"
    )
TRAINABLE_LOCKING = args.trainable_locking
INIT_LOCK_STRENGTH = args.locking_strength
TRAINABLE_COUPLING = args.trainable_coupling
INIT_COUPLING_STRENGTH = args.coupling_strength

# A constant to record where the coupling weight starts in the trainable_analog parameters
ANALOG_WEIGHT_OFFSET = (
    0 if MATRIX_SOLVE else int(TRAINABLE_COUPLING) + int(TRAINABLE_LOCKING)
)

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

LOAD_WEIGHT = args.load_weight
if LOAD_WEIGHT and WEIGHT_INIT:
    print("Ignoring the weight init argument since the weight is loaded")

TESTING = args.test
TEST_DATA = None
WEIGGT_DROP_RATIO = args.weight_drop_ratio

if WEIGGT_DROP_RATIO > 0:
    assert TESTING, "Weight dropping is only for testing"
    assert not WEIGHT_BITS, "Weight dropping is not supported for digital weight"

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
    tags = ["digit_recognitio"] if not TAG else ["digit_recognition", TAG]
    wandb_run = wandb.init(project="obc", config=vars(args), tags=tags)

VECTORIZE_ODETERM = args.vectorize_odeterm


def enumerate_node_pairs(
    n_row=N_ROW, n_col=N_COL
) -> Generator[tuple[int, int, int, int], None, None]:
    """Enumerate all unique node pairs in the network

    Return row, col, row_, col_.
    (row, col) is the axis for the first node and (row_, col_) is for the second.
    row <= row_ and (col < col_ if row == row_).
    """
    for row in range(n_row):
        for col in range(n_col):
            for row_ in range(row, n_row):
                for col_ in range(n_col):
                    if row_ == row and col_ <= col:
                        continue
                    yield row, col, row_, col_


def pattern_to_edge_initialization(pattern: np.ndarray):
    """Map the pattern to the edge initialization

    If two pixels are different, the edge is initialized with -1.0
    Otherwise, the edge is initialized with 1.0
    """

    weight_init = [
        [[[0 for _ in range(N_COL)] for _ in range(N_ROW)] for _ in range(N_COL)]
        for _ in range(N_ROW)
    ]

    for row, col, row_, col_ in enumerate_node_pairs():
        diff = pattern[row, col] - pattern[row_, col_]
        weight_init[row][col][row_][col_] = weight_init[row_][col_][row][col] = (
            -1.0 if diff else 1.0
        )

    return np.array(weight_init)


def edge_init_to_trainable_init(
    weight_init: np.ndarray,
    oscillator_ks: list[list[list[list[Trainable]]]],
    trainable_mgr: TrainableMgr,
) -> tuple[jax.Array, list[jax.Array]]:
    """Convert the edge initialization to trainable initialization"""

    def one_hot_digitize(x: np.ndarray, bins: np.ndarray):
        # Digitize with the nearest bin and then one-hot encode
        # the index is then one-hot encoded
        nearest_bins = np.abs(x[:, :, None] - bins[None, None, :]).argmin(axis=-1)
        one_hot = jax.nn.one_hot(nearest_bins, bins.shape[0])
        return one_hot

    if not FIX_COUPLING_WEIGHT:
        if WEIGHT_BITS is not None:
            weight_choices: list = Cpl_digital.attr_def["k"].attr_type.val_choices

            # normalize the weight choices to between -1 and 1
            weight_choices = np.array(weight_choices)
            weight_init = one_hot_digitize(weight_init, weight_choices)

        for row, col, row_, col_ in enumerate_node_pairs():
            if isinstance(oscillator_ks[row][col][row_][col_], Trainable):
                oscillator_ks[row][col][row_][col_].init_val = weight_init[
                    row, col, row_, col_
                ]

    a_trainable = trainable_mgr.get_initial_vals("analog")
    d_trainable = trainable_mgr.get_initial_vals("digital")

    return (a_trainable, d_trainable)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    loss_fn: Callable,
    data: list[jax.Array],
    gumbel_temp_: jax.Array,  # Somehow causing recompilation if it is float
    hard_gumbel: bool = False,
):
    gumbel_temp = gumbel_temp_[0]
    train_data, val_data = [d[:TRAIN_BZ] for d in data], [d[TRAIN_BZ:] for d in data]
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, *train_data, gumbel_temp, hard_gumbel, L1_NORM_WEIGHT
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    # When validating, always use hard gumbel to force the param to be physical
    val_loss = loss_fn(
        model, *val_data, gumbel_temp, hard_gumbel=True, l1_norm_weight=0
    )
    return model, opt_state, train_loss, val_loss


@eqx.filter_jit
def test_model(
    model: BaseAnalogCkt,
    loss_fn: Callable,
):
    return loss_fn(model, *TEST_DATA, 1.0, True, 0.0)


def plot_evolution(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str,
    gumbel_temp: float,
    hard_gumbel: bool = True,
):
    # python pattern_recog_main.py --n_class 5 --diff_fn periodic_mse  --trans_noise_std 0.025 --steps 64   --bz 48 --seed 666  --num_plot 20 --plot_evolve 4  --no_noiseless_train --pattern_shape 10x6 --weight_init hebbian --weight_bit 1 --uniform_noise
    # plot the 13th
    x_init, args_seed, noise_seed = data[0], data[1], data[2]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0, None, None))(
        time_info, x_init, [], args_seed, noise_seed, gumbel_temp, hard_gumbel
    )
    plot_time = [i for i in range(len(saveat))]

    n_row, n_col = y_raw.shape[0], len(plot_time)
    fig, ax = plt.subplots(
        ncols=len(plot_time), nrows=y_raw.shape[0], figsize=(n_col, n_row * 1.5)
    )
    # fig, ax = plt.subplots(ncols=len(plot_time), nrows=1, figsize=(3, 1.75))
    losses = []
    for i, y in enumerate(y_raw):
        # phase is periodic over 2
        y_readout = y % 2

        di = [d[i : i + 1] for d in data]
        # if i != 13:
        #     continue
        loss = loss_fn(model, *di, gumbel_temp, hard_gumbel, 0)
        losses.append(loss)
        for j, time in enumerate(plot_time):
            y_readout_t = y_readout[time].reshape(N_ROW, N_COL)
            y_readout_t = jnp.abs(y_readout_t - y_readout_t[0, 0])
            # Calculate the phase difference
            phase_diff = jnp.where(y_readout_t > 1, 2 - y_readout_t, y_readout_t)
            ax[i, j].axis("off")
            cmap = "gray_r" if PATTERN_SHAPE == "10x6" else "gray"
            ax[i, j].imshow(phase_diff, cmap=cmap, vmin=0, vmax=1)

            # ax[j].axis("off")
            # ax[j].imshow(phase_diff, cmap="gray_r", vmin=0, vmax=1)
    loss = jnp.mean(jnp.array(losses))
    plt.suptitle(title + f", Loss: {loss:.4f}")

    # Add x-axis to the bottom row
    # plt.rcParams["text.usetex"] = True
    # x_labels = [f"$t_{i}$" for i in plot_time]
    # a: plt.Axes
    # for a, x_label in zip(ax[-1], x_labels):
    # for a, x_label in zip(ax, x_labels):
    #     a.set_title(x_label, y=-0.4, fontsize=25)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if USE_WANDB:
        wandb.log(data={f"{title}_evolution": plt}, commit=False)
        plt.close()
    else:
        plt.savefig(f"{title}-loss_{loss:.4f}.pdf", bbox_inches="tight", dpi=300)
        plt.show()


def initialize_model(
    model_cls: type, weight: tuple[jax.Array, list[jax.Array]], is_stochastic: bool
) -> BaseAnalogCkt | OscillatorNetworkMatrixSolve:
    if MATRIX_SOLVE:
        model: OscillatorNetworkMatrixSolve = model_cls(
            init_coupling=weight[0],
            init_locking=weight[1],
            neighbor_connection=True if CONNECTION == "neighbor" else False,
            is_stochastic=is_stochastic,
            noise_amp=TRANS_NOISE_STD,
            solver=Heun(),
        )
    else:
        model: BaseAnalogCkt = model_cls(
            init_trainable=weight,
            is_stochastic=is_stochastic,
            solver=Heun(),
        )
    return model


def load_model_and_plot(
    model_cls: type,
    best_weight: tuple,
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
    model = initialize_model(model_cls, best_weight, is_stochastic)

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
    best_weight = model.weights()

    gumbel_temp = GUMBEL_TEMP_START
    hard_gumbel = USE_HARD_GUMBEL

    test_losses = []
    val_loss_best = 1e9

    for step, data in zip(range(STEPS), dl(BZ)):

        if TESTING:
            # Test the model: Make sure the seed is different from the training seed
            # so that the data contains different static and transient noise
            # This is for testing more samples than the one tested with training steps.
            test_loss = loss_fn(
                model, *data, gumbel_temp, hard_gumbel=True, l1_norm_weight=0
            )
            test_losses.append(test_loss)
            print(f"Step {step}, Test loss: {test_loss}")

            if USE_WANDB:
                wandb.log(
                    data={
                        f"{log_prefix}_test_loss": test_loss,
                    },
                )

            if step == STEPS - 1:
                print(f"Average test loss: {np.mean(test_losses)}")
                if USE_WANDB:
                    wandb.log(
                        data={
                            f"{log_prefix}_average_test_loss": np.mean(test_losses),
                        },
                    )
            continue

        if step == 0:  # Set up the baseline loss
            train_data = [d[:TRAIN_BZ] for d in data]
            val_data = [d[TRAIN_BZ:] for d in data]
            train_loss = loss_fn(
                model,
                *train_data,
                gumbel_temp,
                hard_gumbel=True,
                l1_norm_weight=L1_NORM_WEIGHT,
            )
            val_loss = loss_fn(
                model, *val_data, gumbel_temp, hard_gumbel=True, l1_norm_weight=0
            )

        else:
            model, opt_state, train_loss, val_loss = make_step(
                model, opt_state, loss_fn, data, jnp.array([gumbel_temp]), hard_gumbel
            )
        test_loss = test_model(model, loss_fn)

        print(
            f"Step {step}, Train loss: {train_loss}, Val loss: {val_loss}, Test loss: {test_loss}, Gumbel temp: {gumbel_temp}"
        )
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            test_loss_best = test_loss
            best_weight = model.weights()

        if USE_WANDB:
            wandb.log(
                data={
                    f"{log_prefix}_train_loss": train_loss,
                    f"{log_prefix}_val_loss": val_loss,
                    f"{log_prefix}_test_loss": test_loss,
                    "gumbel_temp": gumbel_temp,
                },
            )
        gumbel_temp = next_gumbel_temp(step + 1)

    if USE_WANDB:
        # Report the test loss on the iteration with the best validation loss
        wandb.log(
            data={f"{log_prefix}_test_loss_w_best_val": test_loss_best},
        )

    return val_loss_best, best_weight


def new_coupling_weight():
    return (
        trainable_mgr.new_analog() if not WEIGHT_BITS else trainable_mgr.new_digital()
    )


if __name__ == "__main__":

    graph = CDG()
    nodes = [[None for _ in range(N_COL)] for _ in range(N_ROW)]

    if not WEIGHT_BITS:
        cpl_type = Coupling
    else:
        cpl_type = Cpl_digital

    # Store all the trainable parameters
    oscillator_ks = [
        [[[None for _ in range(N_COL)] for _ in range(N_ROW)] for _ in range(N_COL)]
        for _ in range(N_ROW)
    ]

    if WEIGHT_INIT == "hebbian":
        weight_init = pattern_to_edge_initialization(NUMBERS[0])
        for i in range(1, N_CLASS):
            w = pattern_to_edge_initialization(NUMBERS[i])
            weight_init += w

        weight_init /= N_CLASS
    elif WEIGHT_INIT == "random":
        weight_init = np.random.uniform(
            low=-1, high=1, size=(N_ROW, N_COL, N_ROW, N_COL)
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

    for row, col, row_, col_ in enumerate_node_pairs():
        if CONNECTION == "all" or (
            CONNECTION == "neighbor"
            and ((row_ == row + 1 and col_ == col) or (row_ == row and col_ == col + 1))
        ):
            if FIX_COUPLING_WEIGHT:
                edge = cpl_type(k=float(weight_init[row][col][row_][col_]))
            else:
                new_k = new_coupling_weight()
                edge = cpl_type(k=new_k)
                oscillator_ks[row][col][row_][col_] = oscillator_ks[row_][col_][row][
                    col
                ] = new_k
            graph.connect(edge, nodes[row][col], nodes[row_][col_])

    # flatten the nodes for readout
    nodes_flat = [node for row in nodes for node in row]

    rec_circuit_class = (
        OptCompiler().compile(
            "rec",
            graph,
            obc_spec,
            trainable_mgr=trainable_mgr,
            readout_nodes=nodes_flat,
            normalize_weight=False,
            do_clipping=False,
            vectorize=VECTORIZE_ODETERM,
        )
        if not MATRIX_SOLVE
        else OscillatorNetworkMatrixSolve
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
            mapping_fn=(
                None if MATRIX_SOLVE else rec_circuit_class.cdg_to_initial_states
            ),
            snp_prob=SNP_PROB,
            gauss_std=GAUSS_STD,
            uniform_noise=UNIFORM_NOISE,
        )
        loss_fn = partial(
            pattern_reconstruction_loss,
            time_info=time_info,
            diff_fn=diff_fn,
            weight_offset=ANALOG_WEIGHT_OFFSET,
        )
    elif TASK == "rand-to-many":

        dl = partial(dataloader, n_node=N_NODE)
        loss_fn = partial(
            min_rand_reconstruction_loss,
            time_info=time_info,
            diff_fn=diff_fn,
            N_CLASS=N_CLASS,
        )
    # Store the random state and restore later
    random_state = np.random.get_state()

    # Generate the testing data with fixed seed
    np.random.seed(TEST_SEED)
    TEST_DATA = next(dl(TEST_BZ))
    np.random.set_state(random_state)

    if MATRIX_SOLVE:
        if WEIGHT_BITS:
            raise NotImplementedError(
                "Digital weight is not supported for matrix solve"
            )
        if not TRAINABLE_LOCKING:
            raise NotImplementedError(
                "Locking strength must be trainable for matrix solve"
            )
        n_node = N_ROW * N_COL
        if LOAD_WEIGHT:
            weights = jnp.load(LOAD_WEIGHT)
            trainable_init = (weights["new_coupling_weight"], weights["locking_weight"])
            print(trainable_init)
        else:
            trainable_init = (
                jnp.array(weight_init),
                INIT_LOCK_STRENGTH,
            )
    else:
        if not LOAD_WEIGHT:
            trainable_init = edge_init_to_trainable_init(
                weight_init, oscillator_ks, trainable_mgr
            )
        else:
            weights = jnp.load(LOAD_WEIGHT)
            trainable_init = (weights["analog"], weights["digital"])

        if WEIGGT_DROP_RATIO:
            # Set the smallest weights to zero
            analog_weight = trainable_init[0]
            weight_drop_mask = np.abs(analog_weight) <= np.quantile(
                np.abs(analog_weight), WEIGGT_DROP_RATIO
            )
            dropped_analog_weight = jnp.where(weight_drop_mask, 0.0, analog_weight)
            trainable_init = (dropped_analog_weight, trainable_init[1])

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

        model = initialize_model(rec_circuit_class, trainable_init, False)
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
    model = initialize_model(rec_circuit_class, best_weight, True)

    best_loss, best_weight = train(model, loss_fn, dl, "tran_noisy")
    print(f"\tFine-tune Best Loss: {best_loss}")
    print(f"Fine-tune Best Weights: {best_weight}")

    if args.save_weight:
        if MATRIX_SOLVE:
            jnp.savez(
                args.save_weight,
                coupling_weight=best_weight[0],
                locking_weight=best_weight[1],
            )
        else:
            jnp.savez(args.save_weight, analog=best_weight[0], digital=best_weight[1])

    # Model after fine-tune
    load_model_and_plot(
        model_cls=rec_circuit_class,
        best_weight=best_weight,
        is_stochastic=True,
        loss_fn=loss_fn,
        data=plot_data,
        title="Noisy (after fine-tune)",
    )

    # Save the histogram of the coupling strength
    if MATRIX_SOLVE or (not WEIGHT_BITS):
        coupling_weight = best_weight[0][ANALOG_WEIGHT_OFFSET:]
        plt.hist(coupling_weight.flatten(), bins=20)
        plt.title("Coupling strength histogram")
        if USE_WANDB:
            wandb.log(data={"coupling_strength_hist": wandb.Image(plt)})
        else:
            plt.show()
    wandb.finish()
