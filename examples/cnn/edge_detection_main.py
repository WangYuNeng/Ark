if True:  # Temporarily solution to avoid the libomp.dylib error
    from edge_detection_dataloader import TestDataLoader, TrainDataLoader

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
from edge_detection_parser import args
from jaxtyping import PyTree
from spec import (
    FlowE,
    IdealV,
    Inp,
    MapE,
    Out,
    Vm,
    fEm_1p,
    mm_cnn_spec,
    saturation,
    saturation_diffpair,
)

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.trainable import TrainableMgr

mgr = TrainableMgr()

SEED = args.seed
np.random.seed(SEED)

STEPS = args.steps

OPTIMIZER = args.optimizer
LEARNING_RATE = args.lr
optim = getattr(optax, OPTIMIZER)(LEARNING_RATE)

BZ = args.bz
train_loader, test_loader = TrainDataLoader(BZ), TestDataLoader(2)
N_ROW, N_COL = train_loader.image_shape()

WEIGHT_INIT = args.weight_init
MM_NODE = args.mismatched_node
MM_EDGE = args.mismatched_edge
ACTIVATION = args.activation

PLOT_EVOLVE = args.plot_evolve
END_TIME = args.end_time
N_TIME_POINTS = args.n_time_points
if PLOT_EVOLVE != 0:
    saveat = np.linspace(0, END_TIME, PLOT_EVOLVE)
else:
    saveat = [END_TIME]

time_info = TimeInfo(
    t0=0,
    t1=END_TIME,
    dt0=END_TIME / N_TIME_POINTS,
    saveat=saveat,
)

USE_WANDB = args.wandb
if USE_WANDB:
    wandb_run = wandb.init(project="cnn", config=vars(args), tags=["edge-detection"])


def create_cnn(
    nrows: int,
    ncols: int,
    v_nt: NodeType,
    flow_et: EdgeType,
    A_mat: np.array,
    B_mat: np.array,
    bias: int,
    saturation_fn: Callable,
    weight_sharing: bool = True,
):
    """Create a CNN with nrows * ncols nodes

    A_mat, B_mat: 3x3 matrices
    bias: bias for the v nodes
    """

    graph = CDG()
    # Create nodes
    if v_nt == IdealV:
        vs = [
            [v_nt(z=mgr.new_analog(bias)) for _ in range(ncols)] for _ in range(nrows)
        ]
    elif v_nt == Vm:
        vs = [
            [v_nt(z=mgr.new_analog(bias), mm=1.0) for _ in range(ncols)]
            for _ in range(nrows)
        ]
    inps = [[Inp() for _ in range(ncols)] for _ in range(nrows)]
    outs = [[Out(act=saturation_fn) for _ in range(ncols)] for _ in range(nrows)]

    # Create shared trainable attributes
    if weight_sharing:
        A_mat_var = [[mgr.new_analog(val) for val in row] for row in A_mat]
        B_mat_var = [[mgr.new_analog(val) for val in row] for row in B_mat]

    # Create edges
    # All v nodes connect from self, and connect to output
    # in/output node in the corner
    # -> connect v node in that position and 3 neighbors v nodes
    # in/output node on the edge
    # -> connect the v node in that position and 5 neighbors v nodes
    # in/output node in the middle
    # -> connect the v node in that position and 8 neighbors v nodes
    for row_id in range(nrows):
        for col_id in range(ncols):
            v = vs[row_id][col_id]
            inp = inps[row_id][col_id]
            out = outs[row_id][col_id]
            v.set_init_val(val=0.0, n=0)
            graph.connect(MapE(), v, v)
            graph.connect(MapE(), v, out)

            for row_offset in [-1, 0, 1]:
                for col_offset in [-1, 0, 1]:
                    if row_id + row_offset < 0 or row_id + row_offset >= nrows:
                        continue
                    if col_id + col_offset < 0 or col_id + col_offset >= ncols:
                        continue

                    if weight_sharing:
                        in_edge_weight = B_mat_var[row_offset + 1][col_offset + 1]
                        out_edge_weight = A_mat_var[row_offset + 1][col_offset + 1]
                    else:
                        in_edge_weight = mgr.new_analog(
                            B_mat[row_offset + 1, col_offset + 1]
                        )
                        out_edge_weight = mgr.new_analog(
                            A_mat[row_offset + 1, col_offset + 1]
                        )

                    graph.connect(
                        flow_et(g=in_edge_weight),
                        inp,
                        vs[row_id + row_offset][col_id + col_offset],
                    )
                    graph.connect(
                        flow_et(g=out_edge_weight),
                        out,
                        vs[row_id + row_offset][col_id + col_offset],
                    )
    return vs, inps, outs, graph


def mse_loss(
    model: BaseAnalogCkt,
    x: jnp.ndarray,
    args_seed: jnp.array,
    noise_seed: jnp.array,
    y: jnp.ndarray,
    activation: Callable,
):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0))(
        time_info, x, [], args_seed, noise_seed
    )
    y_end_readout = activation(y_raw[:, -1, :])
    return jnp.mean(jnp.square(y_end_readout - y))


def plot_evolution(
    model: BaseAnalogCkt,
    activation: Callable,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str,
):
    x_init, args_seed, noise_seed = data[0], data[1], data[2]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0))(
        time_info, x_init, [], args_seed, noise_seed
    )
    plot_time = [i for i in range(len(saveat))]

    p_rows, p_cols = y_raw.shape[0], len(plot_time)
    fig, ax = plt.subplots(
        ncols=len(plot_time),
        nrows=y_raw.shape[0],
        figsize=(p_cols, p_rows * 1.75),
    )
    losses = []
    for i, y in enumerate(y_raw):
        # phase is periodic over 2
        y_readout = activation(y)

        for j, time in enumerate(plot_time):
            y_readout_t = y_readout[time].reshape(N_ROW, N_COL)
            ax[i, j].axis("off")
            ax[i, j].imshow(y_readout_t, cmap="gray", vmin=-1, vmax=1)

        di = [d[i : i + 1] for d in data]
        losses.append(loss_fn(model, *di))
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
    activation: Callable,
    best_weight: tuple[jax.Array, list[jax.Array]],
    is_stochastic: bool,
    loss_fn: Callable,
    data: list[jax.Array],
    title: str = None,
):
    """Plot the evolution of the oscillator phase"""
    if PLOT_EVOLVE == 0:
        return

    model: BaseAnalogCkt = model_cls(
        init_trainable=best_weight,
        is_stochastic=is_stochastic,
        solver=Heun(),
    )

    plot_evolution(model, activation, loss_fn, data, title)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    loss_fn: Callable,
    data: list[jax.Array],
):
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, train_loss


def train(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    train_dl: Generator,
    test_dl: Generator,
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    loss_best = 1e9

    # Have testing set in the training loop for loggin simplicity...
    # Don't use any information from the testing set for training
    # TODO: Add validation split to better monitor overfitting

    for step in range(STEPS):

        losses = []
        for data in train_dl:
            if step == 0:  # Set up the baseline loss
                train_loss = loss_fn(model, *data)
                losses.append(train_loss)
            else:
                model, opt_state, train_loss = make_step(
                    model, opt_state, loss_fn, data
                )
                losses.append(train_loss)

        train_loss = jnp.mean(losses)

        losses = []
        for data in test_dl:
            val_loss = loss_fn(model, *data)
            losses.append(val_loss)
        test_loss = jnp.mean(losses)

        print(f"Step {step}, Train loss: {train_loss}, Test loss: {test_loss}")

        if train_loss < loss_best:
            loss_best = test_loss
            best_weight = (model.a_trainable.copy(), model.d_trainable.copy())

        if USE_WANDB:
            wandb.log(
                data={
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                },
            )

    return loss_best, best_weight


if __name__ == "__main__":

    if WEIGHT_INIT == "edge-detection":
        A_mat = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        B_mat = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        bias = -0.5
    elif WEIGHT_INIT == "random":
        A_mat = np.random.rand(3, 3)
        B_mat = np.random.rand(3, 3)
        bias = np.random.rand()

    if MM_NODE:
        v_nt = Vm
    else:
        v_nt = IdealV

    if MM_EDGE:
        flow_et = fEm_1p
    else:
        flow_et = FlowE

    if ACTIVATION == "ideal":
        activation_fn = saturation
    elif ACTIVATION == "diffpair":
        activation_fn = saturation_diffpair

    vs, inps, outs, graph = create_cnn(
        N_ROW, N_COL, v_nt, flow_et, A_mat, B_mat, bias, activation_fn
    )

    # Flatten the nodes for readout
    vs_flat = [v for row in vs for v in row]

    cnn_ckt_class = OptCompiler().compile(
        "cnn",
        graph,
        mm_cnn_spec,
        trainable_mgr=mgr,
        readout_nodes=vs_flat,
        normalize_weight=False,
        do_clipping=False,
    )
    train_loader.set_cnn_info(inps, graph, cnn_ckt_class)
    test_loader.set_cnn_info(inps, graph, cnn_ckt_class)

    loss_fn = partial(mse_loss, activation=activation_fn)

    trainable_init = (mgr.get_initial_vals("analog"), mgr.get_initial_vals("digital"))
    plot_data = next(iter(test_loader))

    load_model_and_plot(
        model_cls=cnn_ckt_class,
        activation=activation_fn,
        best_weight=trainable_init,
        is_stochastic=False,
        loss_fn=loss_fn,
        data=plot_data,
        title="Before training",
    )

    model: BaseAnalogCkt = cnn_ckt_class(
        init_trainable=trainable_init, is_stochastic=False, solver=Heun()
    )

    loss_best, best_weight = train(model, mse_loss, train_loader, test_loader)
