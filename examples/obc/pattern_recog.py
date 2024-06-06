import equinox as eqx
import jax
import jax.numpy as jnp
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

trainable_mgr = TrainableMananger()

graph = CDG()

N_NODE = 3
PATTERN = [0, 1]  # phase difference relative to the first node

y_true = jnp.array(PATTERN)
nodes = []

for i in range(N_NODE):
    node = Osc_modified(
        lock_strength=1.2,
        cpl_strength=2,
        lock_fn=locking_fn,
        osc_fn=coupling_fn,
    )
    self_edge = Coupling(k=1)
    graph.connect(self_edge, node, node)
    nodes.append(node)

    for j in range(i):
        edge = Coupling(k=trainable_mgr.new_var())
        graph.connect(edge, node, nodes[j])


def dataloader(batch_size: int):
    while True:
        x_init_states = np.random.rand(batch_size, N_NODE)
        # x_init_states = np.array([0.3, 0.3, 0.3] * batch_size).reshape(
        #     batch_size, N_NODE
        # )
        yield jnp.array(x_init_states)


xor_circuit_class = OptCompiler().compile(
    "xor",
    graph,
    obc_spec,
    readout_nodes=nodes,
    normalize_weight=False,
    do_clipping=False,
)


N_CYCLES = 2

time_info = TimeInfo(
    t0=0, t1=T * N_CYCLES, dt0=T / 50, saveat=jnp.linspace(0, T * N_CYCLES, 100)
)


LEARNING_RATE = 1e-1
optim = optax.adamw(LEARNING_RATE)


def normalize_angular_diff_loss(x: jax.Array, y: jax.Array):
    return jnp.sin(jnp.pi * ((x - y) / 2 % 1))


# Plot to test the loss function
# x = jnp.linspace(-3, 3, 600)
# y = jnp.zeros_like(x)
# plt.plot(x, normalize_angular_diff_loss(x, y))
# plt.show()


def loss(model: BaseAnalogCkt, x: jax.Array):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(time_info, x, [], 0, 0)
    y_readout = y_raw[:, -1, :]
    # Compare the phase relative to the first node with the PATTERN
    y_rel = y_readout[:, 1:] - y_readout[:, 0].repeat(2).reshape(-1, 2)
    loss_val = normalize_angular_diff_loss(y_rel, y_true)
    return jnp.mean(loss_val)


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    x: jax.Array,
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


steps = 12
bz = 128

# losses = []
# fig, ax = plt.subplots(2, 2)

# for i, trainable_init in enumerate(
#     [[-1, -1, 0], [-1, 0, -1], [0, -1, -1], [-1, -1, -1]]
# ):
#     # for trainable_init in [[0, -1, -1]]:
#     model: BaseAnalogCkt = xor_circuit_class(
#         init_trainable=jnp.array(trainable_init),
#         is_stochastic=False,
#         solver=Heun(),
#     )
#     for _, x_init in zip(range(1), dataloader(bz)):
#         y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(
#             time_info, x_init, [], 0, 0
#         )
#         y_readout = y_raw[:, -1, :]
#         # Compare the phase relative to the first node with the PATTERN
#         y_rel = y_readout[:, 1:] - y_readout[:, 0].repeat(2).reshape(-1, 2)
#         loss_val = normalize_angular_diff_loss(y_rel, y_true)
#         t = y_raw[10]
#         # for t in y_raw:
#         ax[i // 2, i % 2].plot(time_info.saveat, t[:, 0], label="Node 0")
#         ax[i // 2, i % 2].plot(time_info.saveat, t[:, 1], label="Node 1")
#         ax[i // 2, i % 2].plot(time_info.saveat, t[:, 2], label="Node 2")
#         ax[i // 2, i % 2].set_title(f"Trainable: {trainable_init}")
#         ax[i // 2, i % 2].legend()
# plt.title(f"Loss: {loss_val}")

# plt the distribution of y_rel
# plt.hist(y_rel[:, 0], bins=20, alpha=0.5, label="Node 1")
# plt.hist(y_rel[:, 1], bins=20, alpha=0.5, label="Node 2")
# plt.legend()
# plt.show()
#     print(f"Weights: {model.trainable}")
#     print(f"Loss: {jnp.mean(loss_val)}")

# plt.tight_layout()
# plt.legend()
# plt.show()

for seed in range(20):
    np.random.seed(seed)
    train_loss_best = 1e9
    model: BaseAnalogCkt = xor_circuit_class(
        # init_trainable=jnp.array([0, -1, -1]),
        init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
        is_stochastic=True,
        solver=Heun(),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    print(f"Seed {seed}")
    print(f"Weights: {model.trainable}")

    for step, x_init in zip(range(steps), dataloader(bz)):

        model, opt_state, train_loss = make_step(model, opt_state, x_init)
        print(f"\tStep {step}, Loss: {train_loss}")
        if train_loss < train_loss_best:
            train_loss_best = train_loss
            best_weight = model.trainable.copy()

    print(f"\tBest Loss: {train_loss_best}")
    print(f"Best Weights: {best_weight}")
