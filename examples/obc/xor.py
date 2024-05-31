import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from diffrax.solver import Euler, Tsit5
from jaxtyping import Array, PyTree
from spec import FREQ, OFFSET_STD, Coupling, Osc, T, obc_spec
from sympy import *

from ark.ark import ArkCompiler
from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.attribute_type import AnalogAttr, FunctionAttr, Trainable
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SRC, VAR
from ark.specification.specification import CDGSpec

jax.config.update("jax_enable_x64", True)


def locking_fn(x):
    """Injection locking function from [2]
    Modify the leading coefficient to 1.2 has a better outcome
    """
    return 1.2 * 795.8e6 * jnp.sin(2 * jnp.pi * x)


def coupling_fn(x):
    """Coupling function from [2]"""
    return 2 * 795.8e6 * jnp.sin(jnp.pi * x)


graph = CDG()
nodes = [Osc(lock_fn=locking_fn, osc_fn=coupling_fn) for _ in range(3)]
connections = [Coupling(k=Trainable(i)) for i in range(3)]

graph.connect(connections[0], nodes[0], nodes[1])
graph.connect(connections[1], nodes[1], nodes[2])
graph.connect(connections[2], nodes[2], nodes[0])

xor_circuit_class = OptCompiler().compile(
    "xor",
    graph,
    obc_spec,
    readout_nodes=[nodes[0], nodes[1], nodes[2]],
    normalize_weight=False,
    do_clipping=False,
)

time_info = TimeInfo(t0=0, t1=T * 3, dt0=T / 10, saveat=jnp.linspace(0, T * 3, 100))

xor_circuit: BaseAnalogCkt = xor_circuit_class(
    init_trainable=jnp.array(np.random.normal(size=3)),
    is_stochastic=False,
    solver=Tsit5(),
)

LEARNING_RATE = 1e-2
optim = optax.adam(LEARNING_RATE)
nodes[2].set_init_val(0.5, 0)


def dataloader(batch_size: int):
    while True:
        xs = np.random.randint(0, 2, (batch_size, 2))
        y_true = np.logical_xor(xs[:, 0], xs[:, 1])
        x_init_states = []
        for x in xs:
            nodes[0].set_init_val(x[0], 0)
            nodes[1].set_init_val(x[1], 0)
            x_init_states.append(xor_circuit.cdg_to_initial_states(graph))
        yield jnp.array(x_init_states), jnp.array(y_true)


def loss(model: BaseAnalogCkt, x: Array, y_true: Array):
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(time_info, x, [], 0, 0)
    y_pred = y_raw[:, -1, 2]
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


steps = 10
bz = 64
model = xor_circuit

opt_state = optim.init(eqx.filter(model, eqx.is_array))


for step, (x, y_true) in zip(range(steps), dataloader(bz)):
    model, opt_state, train_loss = make_step(model, opt_state, x, y_true)
    print(f"Step {step}, Loss: {train_loss}")
