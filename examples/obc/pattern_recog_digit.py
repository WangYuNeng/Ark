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

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler

jax.config.update("jax_enable_x64", True)


NUMBERS = {
    0: np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
    1: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]]),
    2: np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]]),
    3: np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]),
    4: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    5: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]),
    6: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
    7: np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    8: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
    9: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]]),
}

# Node pattern: relative to the first node
NODE_PATTERNS = [None for _ in NUMBERS.keys()]
for i, pattern in NUMBERS.items():
    pattern_flat = pattern.flatten()
    NODE_PATTERNS[i] = jnp.abs(jnp.array(pattern_flat[1:] - pattern_flat[0]))

N_ROW, N_COL = 6, 3
N_NODE = N_ROW * N_COL
N_EDGE = (N_ROW - 1) * N_COL + (N_COL - 1) * N_ROW
N_CLASS = 1

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
optim = optax.adamw(LEARNING_RATE)

STEPS = 32
BZ = 256
VALIDATION_BZ = 4096
PLOT_BZ = 2


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
    while True:
        x_init_states = np.random.rand(batch_size, N_NODE)
        noise_seed = np.random.randint(0, 2**32 - 1, size=batch_size)
        yield jnp.array(x_init_states), jnp.array(noise_seed)


def normalize_angular_diff_loss(x: jax.Array, y: jax.Array):
    return jnp.sin(jnp.pi * ((x - y) / 2 % 1))


def min_reconstruction_loss(model: BaseAnalogCkt, x: jax.Array, noise_seed: jax.Array):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, 0))(
        time_info, x, [], 0, noise_seed
    )
    y_readout = y_raw[:, -1, :]
    # Compare the phase relative to the first node with the PATTERN
    n_repeat = N_NODE - 1
    y_rel = y_readout[:, 1:] - y_readout[:, 0].repeat(n_repeat).reshape(-1, n_repeat)
    losses = []
    for i in range(N_CLASS):
        losses.append(jnp.mean(normalize_angular_diff_loss(y_rel, NODE_PATTERNS[i])))
    losses = jnp.array(losses)
    # Return the minimum average loss
    return jnp.min(losses)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    x: jax.Array,
    noise_seed: jax.Array,
):
    loss_value, grads = eqx.filter_value_and_grad(min_reconstruction_loss)(
        model, x, noise_seed
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def plot_evolution(
    model: BaseAnalogCkt,
    x_init: jax.Array,
    noise_seed: jax.Array,
    title: str = None,
    plot_time: list[int] = PLOT_TIME,
):
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

        loss = min_reconstruction_loss(model, x_init[i : i + 1], noise_seed[i : i + 1])
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

    for seed in range(10):
        np.random.seed(seed)
        train_loss_best = 1e9

        edge_init = pattern_to_edge_initialization(NUMBERS[0])
        # for i in range(1, N_CLASS):
        #     edge_init += pattern_to_edge_initialization(NUMBERS[i])

        # edge_init /= N_CLASS

        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=jnp.array(edge_init),
            # init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
            is_stochastic=False,
            solver=Heun(),
        )

        #
        plot_test_vec = jnp.array(np.random.rand(PLOT_BZ, N_NODE))
        plot_noise_seed = jnp.array(np.random.randint(0, 2**32 - 1, size=PLOT_BZ))
        plot_evolution(
            model=model,
            x_init=plot_test_vec,
            noise_seed=plot_noise_seed,
            title="Before training",
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        print(f"\n\nSeed {seed}")
        # print(f"Weights: {model.trainable}")

        for step, (x_init, noise_seed) in zip(range(STEPS), dataloader(BZ)):

            model, opt_state, train_loss = make_step(
                model, opt_state, x_init, noise_seed
            )
            # print(f"\tStep {step}, Loss: {train_loss}")
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                best_weight = model.trainable.copy()

        # print(f"\tBest Loss: {train_loss_best}")
        print(f"Best Weights: {best_weight}")

        plot_evolution(
            model=model,
            x_init=plot_test_vec,
            noise_seed=plot_noise_seed,
            title="After training",
        )

        # Test the best model w/o noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=False,
            solver=Heun(),
        )

        for _, (x_init, noise_seed) in zip(range(1), dataloader(VALIDATION_BZ)):
            val_loss = min_reconstruction_loss(model, x_init, noise_seed)
            print(f"Validation Loss w/o noise: {val_loss}")

        # Test the best model w/ noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=True,
            solver=Heun(),
        )
        for _, (x_init, noise_seed) in zip(range(1), dataloader(VALIDATION_BZ)):
            val_loss = min_reconstruction_loss(model, x_init, noise_seed)
            print(f"Validation Loss w/ noise: {val_loss}")

        # Double the weight for the noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight * 2,
            is_stochastic=True,
            solver=Heun(),
        )
        for _, (x_init, noise_seed) in zip(range(1), dataloader(VALIDATION_BZ)):
            val_loss = min_reconstruction_loss(model, x_init, noise_seed)
            print(f"Validation Loss w/ noise (double weight): {val_loss}")

        # Fine-tune the best model w/ noise
        model: BaseAnalogCkt = rec_circuit_class(
            init_trainable=best_weight,
            is_stochastic=True,
            solver=Heun(),
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        train_loss_best = 1e9

        for step, (x_init, noise_seed) in zip(range(STEPS), dataloader(BZ)):
            model, opt_state, train_loss = make_step(
                model, opt_state, x_init, noise_seed
            )
            # print(f"\tFine-tune Step {step}, Loss: {train_loss}")
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

        # Test the best model w/ noise
        for _, (x_init, noise_seed) in zip(range(1), dataloader(VALIDATION_BZ)):
            val_loss = min_reconstruction_loss(model, x_init, noise_seed)
            print(f"Validation Loss w/ noise (fine-tune): {val_loss}")
