if True:  # Temporarily solution to avoid the libomp.dylib error
    from edge_detection_dataloader import (
        MNISTTestDataLoader,
        MNISTTrainDataLoader,
        RandomImgDataloader,
        SimpleShapeDataloader,
    )

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
from tqdm import tqdm

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


WEIGHT_INIT = args.weight_init
MM_NODE = args.mismatched_node
MM_EDGE = args.mismatched_edge
ACTIVATION = args.activation

PLOT_EVOLVE = args.plot_evolve
NUM_PLOT = args.num_plot

STORE_EDGE_DETECTION = args.store_edge_detection
ED_IMG_PATH = args.ed_img_path
DW_SAMPLE = args.downsample

BZ = args.bz
DATASET = args.dataset

LONG_COMPILE_DEMO = args.long_compile_demo

if DATASET == "mnist":
    train_dl = MNISTTrainDataLoader(BZ, downsample=DW_SAMPLE)
    test_dl = MNISTTestDataLoader(BZ, downsample=DW_SAMPLE)
    plot_dl = MNISTTestDataLoader(NUM_PLOT, downsample=DW_SAMPLE)

elif DATASET == "simple":
    train_dl = SimpleShapeDataloader(BZ)
    test_dl = SimpleShapeDataloader(BZ)
    plot_dl = SimpleShapeDataloader(NUM_PLOT, shuffle=False)

elif DATASET == "random":
    img_size = (args.rand_img_size, args.rand_img_size)
    train_dl = RandomImgDataloader(BZ, img_size)
    test_dl = RandomImgDataloader(BZ, img_size)
    plot_dl = RandomImgDataloader(NUM_PLOT, img_size, shuffle=False)
N_ROW, N_COL = train_dl.image_shape()

END_TIME = args.end_time
N_TIME_POINTS = args.n_time_points
if PLOT_EVOLVE != 0:
    # Don't need to plot the initial state (always 0)
    saveat = np.linspace(0, END_TIME, PLOT_EVOLVE, endpoint=True)[1:]
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
    # Create shared trainable attributes
    if weight_sharing:
        A_mat_var = [[mgr.new_analog(val) for val in row] for row in A_mat]
        B_mat_var = [[mgr.new_analog(val) for val in row] for row in B_mat]
        bias_var = mgr.new_analog(bias)

    # Create nodes
    if v_nt == IdealV:
        if not weight_sharing:
            bias_var = mgr.new_analog(bias)
        vs = [[v_nt(z=bias_var) for _ in range(ncols)] for _ in range(nrows)]
    elif v_nt == Vm:
        if not weight_sharing:
            bias_var = mgr.new_analog(bias)
        vs = [[v_nt(z=bias_var, mm=1.0) for _ in range(ncols)] for _ in range(nrows)]
    inps = [[Inp() for _ in range(ncols)] for _ in range(nrows)]
    outs = [[Out(act=saturation_fn) for _ in range(ncols)] for _ in range(nrows)]

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
    y_true = data[3]
    y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0))(
        time_info, x_init, [], args_seed, noise_seed
    )
    plot_time = [i for i in range(len(saveat) + 1)]

    p_rows, p_cols = y_raw.shape[0], len(plot_time) + 1
    fig, ax = plt.subplots(
        ncols=p_cols,
        nrows=p_rows,
        figsize=(p_cols, p_rows * 1.75),
    )
    losses = []
    for i, y in enumerate(y_raw):
        y_readout = activation(y)

        for j, time in enumerate(plot_time):
            y_readout_t = y_readout[time].reshape(N_ROW, N_COL)
            ax[i, j].axis("off")
            ax[i, j].imshow(y_readout_t, cmap="gray_r", vmin=-1, vmax=1)

        # Plot the image under ideal cnn (target)
        ax[i, -1].axis("off")
        ax[i, -1].imshow(
            y_true[i].reshape(N_ROW, N_COL), cmap="gray_r", vmin=-1, vmax=1
        )
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
    activation: Callable,
    opt_state: PyTree,
    loss_fn: Callable,
    data: list[jax.Array],
):
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *data, activation)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, train_loss


def train(
    model: BaseAnalogCkt,
    activation: Callable,
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
        for data in tqdm(train_dl, desc="training"):
            if step == 0:  # Set up the baseline loss
                train_loss = loss_fn(model, *data, activation)
                losses.append(train_loss)
            else:
                model, opt_state, train_loss = make_step(
                    model, activation, opt_state, loss_fn, data
                )
                losses.append(train_loss)

        train_loss = jnp.mean(jnp.array(losses))

        losses = []
        for data in tqdm(test_dl, desc="testing"):
            loss = loss_fn(model, *data, activation)
            losses.append(loss)
        test_loss = jnp.mean(jnp.array(losses))

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


@eqx.filter_jit
def iterate_all_data(model, data_loader, activation_fn):

    imgs = []
    for data in tqdm(data_loader, desc="Generating edge detected images"):
        y_raw = jax.vmap(model, in_axes=(None, 0, None, 0, 0))(
            time_info, data[0], [], data[1], data[2]
        )
        y_end_readout = activation_fn(y_raw[:, -1, :])
        y_imgs = y_end_readout.reshape(-1, N_ROW, N_COL)
        imgs.extend(y_imgs)
    return np.array(imgs)


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

    if MM_EDGE != 0:
        flow_et = fEm_1p
        flow_et.attr_def["g"].rstd = MM_EDGE
    else:
        flow_et = FlowE

    if ACTIVATION == "ideal":
        activation_fn = saturation
    elif ACTIVATION == "diffpair":
        activation_fn = saturation_diffpair

    if DATASET == "random":
        # Create ideal CNN to generate data
        # Using saturation activation function as the target
        # (Although strictly speaking, the "ideal" activation function should
        # not be the diffpair, but in practice diffpair is better)
        vs, inps, outs, graph = create_cnn(
            N_ROW, N_COL, IdealV, FlowE, A_mat, B_mat, bias, activation_fn
        )
        vs_flat = [v for row in vs for v in row]
        ideal_cnn_class = OptCompiler().compile(
            "ideal_cnn",
            graph,
            mm_cnn_spec,
            trainable_mgr=mgr,
            readout_nodes=vs_flat,
            normalize_weight=False,
            do_clipping=False,
            aggregate_args_lines=True,
        )
        trainable_init = (
            mgr.get_initial_vals("analog"),
            mgr.get_initial_vals("digital"),
        )

        model: BaseAnalogCkt = ideal_cnn_class(
            init_trainable=trainable_init, is_stochastic=False, solver=Heun()
        )

        for dl in [train_dl, test_dl, plot_dl]:
            assert isinstance(dl, RandomImgDataloader)
            dl.set_cnn_info(inps, graph, ideal_cnn_class)
            dl.gen_edge_detected_img(model, time_info, activation_fn)

        mgr.reset()

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
        aggregate_args_lines=True,
    )
    train_dl.set_cnn_info(inps, graph, cnn_ckt_class)
    test_dl.set_cnn_info(inps, graph, cnn_ckt_class)
    plot_dl.set_cnn_info(inps, graph, cnn_ckt_class)

    loss_fn = partial(mse_loss, activation=activation_fn)

    trainable_init = (mgr.get_initial_vals("analog"), mgr.get_initial_vals("digital"))
    print(f"Trainable init: {trainable_init}")

    model: BaseAnalogCkt = cnn_ckt_class(
        init_trainable=trainable_init, is_stochastic=False, solver=Heun()
    )

    if LONG_COMPILE_DEMO:
        assert DATASET == "mnist"
        # Just use the input for output image so that we
        # don't need to recollect the ideal edge detected images
        train_dl.load_edge_detected_data(train_dl.images)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        for step in range(10):
            for data in tqdm(train_dl, desc="training"):
                model, opt_state, train_loss = make_step(
                    model, activation_fn, opt_state, mse_loss, data
                )

    if DATASET == "mnist":
        if STORE_EDGE_DETECTION:
            # Iterate over the training and testing data to generate edge detected
            # images with the current cnn templates
            train_dl.shuffle = False
            test_dl.shuffle = False
            train_imgs = iterate_all_data(model, train_dl, activation_fn)
            test_imgs = iterate_all_data(model, test_dl, activation_fn)
            np.savez(ED_IMG_PATH, train=train_imgs, test=test_imgs)
            train_dl.shuffle = True
            test_dl.shuffle = True
            exit()

        ed_imgs = np.load(ED_IMG_PATH)
        train_idl_imgs = ed_imgs["train"]
        test_idl_imgs = ed_imgs["test"]
        train_dl.load_edge_detected_data(train_idl_imgs)
        test_dl.load_edge_detected_data(test_idl_imgs)
        plot_dl.load_edge_detected_data(test_idl_imgs)

    plot_data = next(iter(plot_dl))

    load_model_and_plot(
        model_cls=cnn_ckt_class,
        activation=activation_fn,
        best_weight=trainable_init,
        is_stochastic=False,
        loss_fn=loss_fn,
        data=plot_data,
        title="Before training",
    )

    loss_best, best_weight = train(model, activation_fn, mse_loss, train_dl, test_dl)

    print(f"Best loss: {loss_best}")
    print(f"Best weight: {best_weight}")
    load_model_and_plot(
        model_cls=cnn_ckt_class,
        activation=activation_fn,
        best_weight=best_weight,
        is_stochastic=False,
        loss_fn=loss_fn,
        data=plot_data,
        title="After training",
    )
    wandb.finish()
