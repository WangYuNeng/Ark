from dataclasses import dataclass
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
from sat_utils import BLUE_PHASE, FALSE_PHASE, TRUE_PHASE, create_3sat_graph

import wandb
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr

jax.config.update("jax_enable_x64", True)
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


def plot_results(model: BaseAnalogCkt, data: jax.Array, switches: jax.Array):
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
        model_out = model(ti, data[i], switches, 0, 0).T
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

    graph, nw = create_3sat_graph(2, 2)

    # flatten the var_oscs
    nodes_flat = []
    for var_osc in nw.var_oscs:
        nodes_flat.extend(list(var_osc))

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
    nw.set_var_clause_cpls_args_idx(model=model)
    print(nw.var_clause_cpls_args_idx)
    clauses = [(-1, -1, -1), (-2, -2, -2)]
    switch_arr = jnp.array(nw.clauses_to_switch_array(clauses))
    print("Switch array: ", switch_arr)

    test_traces = np.random.uniform(0, 1, (N_PLOT, len(graph.stateful_nodes)))
    plot_results(model, test_traces, switch_arr)
    model = train(model=model, loss_fn=loss_all_true, dl=data_loader)
    for name, val in zip(
        ["locking_strength", "cpl_strength", "cpl_value"], model.a_trainable
    ):
        print(f"{name}: {val}")
    plot_results(model, test_traces)
