from functools import partial
from typing import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import spec_optimization as opt_spec
from diffrax import Tsit5
from jaxtyping import PyTree

import wandb
from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr

jax.config.update("jax_enable_x64", True)
(FALSE_PHASE, TRUE_PHASE, BLUE_PHASE) = (0, 2 / 3, 4 / 3)
trainable_mgr = TrainableMgr()
optim = optax.adam(learning_rate=1e-1)
time_info = TimeInfo(
    t0=0.0,
    t1=1.0,
    dt0=0.1,
    saveat=[1.0],
)

BZ = 1024
STEPS = 50
N_TOTAL_OSCS = 2 + 6


def locking_3x(x, lock_strength: float):
    return lock_strength * jnp.sin(3 * jnp.pi * x)


def create_3sat_graph(clauses: list[tuple[int, int, int]]):
    """
    Create a 3-SAT graph from the given clauses.

    Args:
        clauses: A list of tuples representing the clauses. Each tuple contains three integers,
                 where each integer is a variable (positive for true, negative for false).
    """

    n_vars = max(abs(var) for clause in clauses for var in clause)

    sat_graph = CDG()

    # Create True, False, Blue oscillators
    f_osc, t_osc, b_osc = (
        opt_spec.FixedSource(phase=FALSE_PHASE),
        opt_spec.FixedSource(phase=TRUE_PHASE),
        opt_spec.FixedSource(phase=BLUE_PHASE),
    )

    # Create varaible oscillators var_oscs[i][0] for -(i+1) and var_oscs[i][1] for +(i+1)
    # var_cpls[i][0] for -(i+1) to Blue, var_cpls[i][1] for +(i+1) to Blue, and var_cpls[i][2]
    # for -(i+1) to +(i+1)
    osc_args = {
        "lock_fn": locking_3x,
        "osc_fn": opt_spec.coupling_fn,
        "lock_strength": trainable_mgr.new_analog(init_val=1.0),
        "cpl_strength": trainable_mgr.new_analog(init_val=1.0),
    }
    cpl_args = {
        "k": trainable_mgr.new_analog(init_val=-1.0),
    }
    var_oscs, var_cpls = [], []
    for _ in range(n_vars):
        oscs = [opt_spec.Osc_modified(**osc_args) for _ in range(2)]
        cpls = [opt_spec.Coupling(**cpl_args) for _ in range(3)]

        # Self connection for locking
        for osc in oscs:
            sat_graph.connect(opt_spec.Coupling(k=1.0), osc, osc)

        # Connect variable oscillators to Blue
        sat_graph.connect(cpls[0], b_osc, oscs[0])
        sat_graph.connect(cpls[1], b_osc, oscs[1])

        # Negative coupling between pos and neg variable oscillators
        # Two variable oscillators should be out of phase
        sat_graph.connect(cpls[2], oscs[0], oscs[1])
        var_oscs.append(oscs)
        var_cpls.append(cpls)

    # Create clause oscillators and connect them to the variable oscillators
    clause_oscs, clause_cpls = [], []
    clause_var_conn_idx = [0, 3, 5]
    for clause in clauses:
        oscs = [opt_spec.Osc_modified(**osc_args) for _ in range(6)]
        cpls = [opt_spec.Coupling(**cpl_args) for _ in range(13)]

        # Self connection for locking
        for osc in oscs:
            sat_graph.connect(opt_spec.Coupling(k=1.0), osc, osc)

        # Connect internal clause oscillators
        sat_graph.connect(cpls[0], oscs[0], oscs[1])
        sat_graph.connect(cpls[1], oscs[1], oscs[2])
        sat_graph.connect(cpls[2], oscs[2], oscs[3])
        sat_graph.connect(cpls[3], oscs[2], oscs[4])
        sat_graph.connect(cpls[4], oscs[4], oscs[5])
        sat_graph.connect(cpls[5], t_osc, oscs[0])
        sat_graph.connect(cpls[6], t_osc, oscs[1])
        sat_graph.connect(cpls[7], t_osc, oscs[3])
        sat_graph.connect(cpls[8], t_osc, oscs[5])
        sat_graph.connect(cpls[9], f_osc, oscs[4])

        # Connect clause oscillators to variable oscillators
        for i, (clause_osc_id, var) in enumerate(zip(clause_var_conn_idx, clause)):
            cpl_idx = 10 + i
            var_idx = abs(var) - 1
            is_pos = var > 0
            sat_graph.connect(
                cpls[cpl_idx],
                oscs[clause_osc_id],
                var_oscs[var_idx][is_pos],
            )
        clause_oscs.append(oscs)
        clause_cpls.append(cpls)

    base_oscs = [f_osc, t_osc, b_osc]

    base_oscs: list[CDGNode]
    var_oscs: list[list[CDGNode]]
    clause_oscs: list[list[CDGNode]]

    return sat_graph, base_oscs, var_oscs, clause_oscs, var_cpls, clause_cpls


def loss_all_true(model: BaseAnalogCkt, x: jax.Array):
    """
    Loss function for the 3-SAT problem. The loss is the sum of the squared differences between
    the output of the model and the target value (1 for True, 0 for False).
    """
    # Get the output of the mode
    y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(time_info, x, [], 0, 0)
    sine_y = jnp.sin(y_raw * jnp.pi)

    # Calculate the loss
    all_true_arr = jnp.sin(jnp.array([FALSE_PHASE, TRUE_PHASE]) * jnp.pi)
    loss = jnp.mean((sine_y - all_true_arr) ** 2)
    return loss


@eqx.filter_jit
def make_step(
    model: BaseAnalogCkt,
    opt_state: PyTree,
    loss_fn: Callable,
    data: jax.Array,
):
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, train_loss


def data_loader(batch_size: int):
    """
    Data loader for the 3-SAT problem. Generates random input data for the model.
    """
    while True:
        # Generate random input data
        x = np.random.uniform(0, 1, (batch_size, N_TOTAL_OSCS))
        # Convert to jax array
        x = jnp.array(x)
        yield x


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for step, data in zip(range(STEPS), dl(BZ)):

        model, opt_state, train_loss = make_step(model, opt_state, loss_fn, data)

        print(f"Step {step}, Train loss: {train_loss}")

    return model


N_PLOT = 3


def plot_results(model: BaseAnalogCkt, data: jax.Array):
    """
    Plot the results of the model.
    """
    ti = TimeInfo(
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        saveat=jnp.arange(0, 1.0, 0.01),
    )

    figs, axes = plt.subplots(nrows=N_PLOT, ncols=1, figsize=(10, 5))
    for i in range(N_PLOT):
        axes[i].set_title(f"Oscillator {i}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Phase")
        model_out = model(ti, data[i], [], 0, 0).T
        for trace in model_out:
            axes[i].plot(
                ti.saveat,
                trace,
            )

        # Use 3 colores to represent the [FALSE, TRUE, BLUE] phases
        # FALSE -> red, TRUE -> green, BLUE -> blue

        for phase in [FALSE_PHASE, TRUE_PHASE, BLUE_PHASE]:
            axes[i].axhline(
                y=phase,
                color="r",
                linestyle="--",
            )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(428)
    clauses = [(1, 1, 1)]
    graph, _, var_oscs, claus_oscs, _, _ = create_3sat_graph(clauses)

    # flatten the var_oscs
    nodes_flat = []
    for var_osc in var_oscs:
        nodes_flat.extend(var_osc)

    ckt_class = OptCompiler().compile(
        prog_name="3sat",
        cdg=graph,
        cdg_spec=opt_spec.obc_spec,
        trainable_mgr=trainable_mgr,
        readout_nodes=nodes_flat,
        normalize_weight=False,
        do_clipping=False,
        vectorize=True,
    )

    init_weight = trainable_mgr.get_initial_vals()
    model: BaseAnalogCkt = ckt_class(
        init_trainable=init_weight,
        is_stochastic=False,
        solver=Tsit5(),
    )
    test_traces = np.random.uniform(0, 1, (N_PLOT, N_TOTAL_OSCS))
    for name, val in zip(
        ["locking_strength", "cpl_strength", "cpl_value"], model.a_trainable
    ):
        print(f"{name}: {val}")
    plot_results(model, test_traces)
    model = train(model=model, loss_fn=loss_all_true, dl=data_loader)
    for name, val in zip(
        ["locking_strength", "cpl_strength", "cpl_value"], model.a_trainable
    ):
        print(f"{name}: {val}")
    plot_results(model, test_traces)
