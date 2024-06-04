import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax.solver import Tsit5
from jaxtyping import Array, PyTree
from spec import Coupling, Osc, T, obc_spec
from sympy import *

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.attribute_def import AttrDef
from ark.specification.attribute_type import AnalogAttr, FunctionAttr, Trainable
from ark.specification.cdg_types import NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR

jax.config.update("jax_enable_x64", True)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
np.random.seed(seed)


class TrainableMananger:

    def __init__(self):
        self.idx = -1

    def new_var(self):
        self.idx += 1
        return Trainable(self.idx)


def locking_fn(x, lock_strength: float):
    """Injection locking function from [2]
    Modify the leading coefficient to 1.2 has a better outcome
    """
    return lock_strength * 795.8e6 * jnp.sin(2 * jnp.pi * x)


def coupling_fn(x, cpl_strength: float):
    """Coupling function from [2]"""
    return cpl_strength * 795.8e6 * jnp.sin(jnp.pi * x)


Osc_modified = NodeType(
    "Osc_modified",
    bases=Osc,
    attrs={
        "order": 1,
        "attr_def": {
            "lock_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "osc_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "lock_strength": AttrDef(attr_type=AnalogAttr((-10, 10))),
            "cpl_strength": AttrDef(attr_type=AnalogAttr((-10, 10))),
        },
    },
)

modified_cp_src = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    SRC,
    -EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST), SRC.cpl_strength),
)

modified_cp_dst = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    DST,
    -EDGE.k * DST.osc_fn(VAR(SRC) - VAR(DST), DST.cpl_strength),
)

modified_cp_self = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    SELF,
    -EDGE.k * SRC.lock_fn(VAR(SRC), SRC.lock_strength),
)

obc_spec.add_production_rules([modified_cp_src, modified_cp_dst, modified_cp_self])

graph = CDG()
trainable_mgr = TrainableMananger()


def mk_edge():
    return Coupling(k=trainable_mgr.new_var())


def mk_node():
    node = Osc_modified(
        lock_fn=locking_fn,
        osc_fn=coupling_fn,
        lock_strength=trainable_mgr.new_var(),
        cpl_strength=trainable_mgr.new_var(),
    )
    node.set_init_val(0.5, 0)
    self_edge = mk_edge()
    graph.connect(self_edge, node, node)
    return node


if __name__ == "__main__":

    N_LAYER = 2
    layer_node = [[] for _ in range(N_LAYER)]
    layer_edge = [[] for _ in range(N_LAYER)]
    # Create node for each layer
    for i in range(N_LAYER):
        layer_node[i] = [mk_node() for _ in range(2)]

    # output_node = mk_node()

    # Fully connect the layers
    for i in range(N_LAYER - 1):
        layer_edge[i] = [mk_edge() for _ in range(4)]
        graph.connect(layer_edge[i][0], layer_node[i][0], layer_node[i + 1][0])
        graph.connect(layer_edge[i][1], layer_node[i][0], layer_node[i + 1][1])
        graph.connect(layer_edge[i][2], layer_node[i][1], layer_node[i + 1][0])
        graph.connect(layer_edge[i][3], layer_node[i][1], layer_node[i + 1][1])

    # Connect the last layer to the output node
    # output_edge = [mk_edge() for _ in range(2)]
    # graph.connect(output_edge[0], layer_node[N_LAYER - 1][0], output_node)
    # graph.connect(output_edge[1], layer_node[N_LAYER - 1][1], output_node)

    xor_circuit_class = OptCompiler().compile(
        "xor",
        graph,
        obc_spec,
        readout_nodes=layer_node[N_LAYER - 1],
        normalize_weight=False,
        do_clipping=False,
    )

    N_CYCLES = 10

    time_info = TimeInfo(
        t0=0, t1=T * N_CYCLES, dt0=T / 10, saveat=jnp.linspace(0, T * N_CYCLES, 100)
    )

    xor_circuit: BaseAnalogCkt = xor_circuit_class(
        init_trainable=jnp.array(np.random.normal(size=trainable_mgr.idx + 1)),
        is_stochastic=False,
        solver=Tsit5(),
    )

    LEARNING_RATE = 1e-2
    optim = optax.adam(LEARNING_RATE)

    def dataloader(batch_size: int):
        while True:
            xs = np.random.randint(0, 2, (batch_size, 2))
            y_true = np.logical_xor(xs[:, 0], xs[:, 1])
            x_init_states = []
            for x in xs:
                layer_node[0][0].set_init_val(x[0], 0)
                layer_node[0][1].set_init_val(x[1], 0)
                x_init_states.append(xor_circuit.cdg_to_initial_states(graph))
            yield jnp.array(x_init_states), jnp.array(y_true)

    def loss(model: BaseAnalogCkt, x: Array, y_true: Array):
        y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(
            time_info, x, [], 0, 0
        )
        y_pred = jnp.abs(y_raw[:, -1, 0] - y_raw[:, -1, 1])
        return jnp.mean((y_pred - y_true) ** 2)

    @eqx.filter_jit
    def make_step(
        model: BaseAnalogCkt,
        opt_state: PyTree,
        x: jax.Array,
        y_true: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y_true)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    steps = 5
    bz = 64
    model = xor_circuit

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    import matplotlib.pyplot as plt

    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    x_init_states = []
    for x in xs:
        layer_node[0][0].set_init_val(x[0], 0)
        layer_node[0][1].set_init_val(x[1], 0)
        x_init_states.append(xor_circuit.cdg_to_initial_states(graph))

    y_preds = jax.vmap(model, in_axes=(None, 0, None, None, None))(
        time_info, jnp.array(x_init_states), [], 0, 0
    )

    fig, axs = plt.subplots(2, 2)
    for i, y_pred in enumerate(y_preds):
        x = xs[i]
        axs[i // 2, i % 2].plot(time_info.saveat, y_pred[:, 0], label="Osc0")
        axs[i // 2, i % 2].plot(time_info.saveat, y_pred[:, 1], label="Osc1")
        axs[i // 2, i % 2].set_title(
            f"Input: {x}, XOR: {int(jnp.logical_xor(x[0], x[1]))}"
        )
        axs[i // 2, i % 2].legend()
    plt.suptitle("Oscillator dynamics before optimization")
    plt.tight_layout()
    plt.savefig("xor_before_training.png")
    plt.show()

    for step, (x, y_true) in zip(range(steps), dataloader(bz)):
        model, opt_state, train_loss = make_step(model, opt_state, x, y_true)
        print(f"Step {step}, Loss: {train_loss}")

        # Enumerate 00, 01, 10, 11 to check the output
        y_preds = jax.vmap(model, in_axes=(None, 0, None, None, None))(
            time_info, jnp.array(x_init_states), [], 0, 0
        )
        for x, y_pred in zip(xs, y_preds):
            print(
                f"Node Input: {x}, XOR: {int(jnp.logical_xor(x[0], x[1]))}, Optimized output: {jnp.abs(y_pred[-1,0]-y_pred[-1,1])}"
            )
    fig, axs = plt.subplots(2, 2)

    N_CYCLES = 32

    time_info = TimeInfo(
        t0=0, t1=T * N_CYCLES, dt0=T / 1000, saveat=jnp.linspace(0, T * N_CYCLES, 100)
    )

    for i, y_pred in enumerate(y_preds):
        x = xs[i]
        axs[i // 2, i % 2].plot(time_info.saveat, y_pred[:, 0], label="Osc0")
        axs[i // 2, i % 2].plot(time_info.saveat, y_pred[:, 1], label="Osc1")
        axs[i // 2, i % 2].set_title(
            f"Input: {x}, XOR: {int(jnp.logical_xor(x[0], x[1]))}"
        )
        axs[i // 2, i % 2].legend()
    plt.suptitle("Oscillator dynamics after optimization")
    plt.tight_layout()
    plt.savefig("xor_after_training.png")
    plt.show()
