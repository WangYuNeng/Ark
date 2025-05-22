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
from sat_dataloader import SATDataloader, sat_3var7clauses_data
from sat_loss import loss_w_sol
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

BZ = 128
STEPS = 300


@eqx.filter_jit
def make_step(model: BaseAnalogCkt, opt_state: PyTree, loss_fn: Callable, data):
    train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, train_loss


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    print("Initial trainable params:")
    print(model.a_trainable)
    for step, data in zip(range(STEPS), dl):

        model, opt_state, train_loss = make_step(model, opt_state, loss_fn, data)

        print(f"\nStep {step}, Train loss: {train_loss}")
        print("Trainable params")
        print(model.a_trainable)

    return model


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

    n_plot = data.shape[0]
    figs, axes = plt.subplots(nrows=n_plot, ncols=1, figsize=(10, 5))
    for i in range(n_plot):
        axes[i].set_title(f"Oscillator {i}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Phase")
        model_out = model(ti, data[i], switches[i], 0, 0).T
        for osc_id, trace in enumerate(model_out):
            sign = "-" if osc_id % 2 == 0 else "+"
            label = f"{sign}var{osc_id // 2 + 1}"
            axes[i].plot(ti.saveat, trace, label=label)

        # Use 3 colores to represent the [FALSE, TRUE, BLUE] phases
        # FALSE -> red, TRUE -> green, BLUE -> blue

        for phase in [FALSE_PHASE, TRUE_PHASE, BLUE_PHASE]:
            axes[i].axhline(
                y=phase,
                color="r",
                linestyle="--",
            )
        axes[i].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(428)

    graph, nw = create_3sat_graph(n_vars=3, n_clauses=7, trainable_mgr=trainable_mgr)

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
        do_clipping=True,
        vectorize=True,
    )

    init_weight = trainable_mgr.get_initial_vals()
    model: BaseAnalogCkt = ckt_class(
        init_trainable=init_weight,
        is_stochastic=False,
        solver=Tsit5(),
    )
    nw.set_var_clause_cpls_args_idx(model=model)

    sat_probs, sat_solutions = sat_3var7clauses_data()
    dataloader = SATDataloader(BZ, sat_probs, nw, sat_solutions)
    init_states, switches, sol = dataloader.__iter__().__next__()

    print(sol[:2])
    print(jnp.sin(sol[:2] * jnp.pi * TRUE_PHASE))
    plot_results(model, init_states[:2], switches[:2])

    loss_fn = partial(loss_w_sol, time_info=time_info)

    model = train(model=model, loss_fn=loss_fn, dl=dataloader)
    print(sol[:2])
    plot_results(model, init_states[:2], switches[:2])
