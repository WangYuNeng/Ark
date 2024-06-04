import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax import Tsit5
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

trainable_mgr = TrainableMananger()

graph = CDG()

N_IN_NODE = 4
N_OUT_NODE = 2
N_INTERNAL_NODE = 2
N_NODE = N_IN_NODE + N_INTERNAL_NODE + N_OUT_NODE
nodes = []

for i in range(N_NODE):
    node = Osc_modified(
        lock_fn=locking_fn,
        osc_fn=coupling_fn,
        lock_strength=1,
        cpl_strength=1,
    )
    self_edge = Coupling(k=1)
    graph.connect(self_edge, node, node)
    nodes.append(node)

for i in range(2):
    edge = Coupling(k=1, switchable=True)
    graph.connect(edge, nodes[i * 2], nodes[i * 2 + 1])

for i in range(N_INTERNAL_NODE):
    node_id = N_IN_NODE + i
    for in_id in range(N_IN_NODE):
        edge = Coupling(k=trainable_mgr.new_var())
        graph.connect(edge, nodes[in_id], nodes[node_id])

    for out_id in range(N_OUT_NODE):
        edge = Coupling(k=trainable_mgr.new_var())
        graph.connect(edge, nodes[node_id], nodes[N_IN_NODE + N_INTERNAL_NODE + out_id])

    for other_internal_id in range(N_IN_NODE, node_id):
        edge = Coupling(k=trainable_mgr.new_var())
        graph.connect(edge, nodes[other_internal_id], nodes[node_id])


def dataloader(batch_size: int):
    while True:
        switches = np.random.randint(0, 2, (batch_size, 2))
        y_true = np.logical_xor(switches[:, 0], switches[:, 1])
        x_init_states = np.random.rand(batch_size, N_NODE)
        yield jnp.array(switches), jnp.array(x_init_states), jnp.array(y_true)


def sorted_data(batch_size: int):
    assert batch_size % 4 == 0
    switches = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * (batch_size // 4))
    y_true = np.logical_xor(switches[:, 0], switches[:, 1])
    x_init_states = np.random.rand(batch_size, N_NODE)
    return jnp.array(switches), jnp.array(x_init_states), jnp.array(y_true)


xor_circuit_class = OptCompiler().compile(
    "xor",
    graph,
    obc_spec,
    readout_nodes=[nodes[-2], nodes[-1]],
    normalize_weight=False,
    do_clipping=False,
)

N_CYCLES = 1

time_info = TimeInfo(
    t0=0, t1=T * N_CYCLES, dt0=T / 10, saveat=jnp.linspace(0, T * N_CYCLES, 100)
)


LEARNING_RATE = 1e-3
optim = optax.adam(LEARNING_RATE)


def loss(model: BaseAnalogCkt, switch: jax.Array, x: jax.Array, y_true: jax.Array):
    y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
        time_info, x, switch, 0, 0
    )
    y_pred = jnp.abs(y_raw[:, -1, 0] - y_raw[:, -1, 1])
    return jnp.mean((y_pred - y_true) ** 2)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    switch: jax.Array,
    x: jax.Array,
    y_true: jax.Array,
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, switch, x, y_true)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


steps = 12
bz = 128

losses = []
import matplotlib.pyplot as plt

for seed in range(20):
    np.random.seed(seed)
    train_loss_best = 1e9
    model: BaseAnalogCkt = xor_circuit_class(
        init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
        is_stochastic=False,
        solver=Tsit5(),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # for step, (switch, x_init, y_true) in zip(range(steps), dataloader(bz)):
    #     model, opt_state, train_loss = make_step(
    #         model, opt_state, switch, x_init, y_true
    #     )
    #     # print(f"Step {step}, Loss: {train_loss}")
    #     if train_loss < train_loss_best:
    #         train_loss_best = train_loss

    #     switch, x_init, y_true = sorted_data(bz)
    #     fig, ax = plt.subplots(2, 2, figsize=(16, 4))
    #     for i in range(4):
    #         x_i, switch_i = x_init[0::4], switch[i::4]
    #         y_true_i = y_true[i::4]
    #         y_raw = jax.vmap(model, in_axes=(None, 0, 0, None, None))(
    #             time_info, x_i, switch_i, 0, 0
    #         )
    #         y_pred = jnp.abs(y_raw[:, -1, 0] - y_raw[:, -1, 1])
    #         mse_loss = jnp.mean((y_pred - y_true_i) ** 2)
    #         print(
    #             f"\tSwitch: {switch_i[0]}, Loss: {mse_loss}, Avg_pred: {jnp.mean(y_pred)}"
    #         )
    #         ax[i // 2, i % 2].plot(y_raw[0, :, 0])
    #         ax[i // 2, i % 2].plot(y_raw[0, :, 1])
    #         ax[i // 2, i % 2].set_title(f"Switch: {switch_i[0]}")
    #     plt.tight_layout()
    #     plt.show()
    print(f"Seed {seed}, Best Loss: {train_loss_best}")
    losses.append(train_loss_best)

# plot histogram of losses

plt.hist(losses, bins=20)
plt.show()
