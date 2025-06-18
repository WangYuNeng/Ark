from functools import partial
from typing import Callable, Generator, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import spec_optimization as opt_spec
from diffrax import Tsit5
from jaxtyping import PyTree
from sat_dataloader import SATDataloader, sat_3var7clauses_data, sat_from_cnf_dir
from sat_loss import loss_w_sol
from sat_parser import parser
from sat_utils import (
    BLUE_PHASE,
    FALSE_PHASE,
    TRUE_PHASE,
    create_3sat_graph,
    flatten_nw_stateful_oscillators,
)

import wandb
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr

jax.config.update("jax_enable_x64", True)
args = parser.parse_args()

SEED = args.seed

T1 = args.t1
DT0 = args.dt0

BZ = args.batch_size
STEPS = args.steps
LR = args.lr

TASK = args.task
CNF_DIR: Optional[str] = args.cnf_dir

USE_WANDB = args.wandb
RUN_NAME = args.run_name
TAG = args.tag

N_PLOT = args.n_plots

trainable_mgr = TrainableMgr()
optim = optax.adam(learning_rate=LR)
time_info = TimeInfo(
    t0=0.0,
    t1=T1,
    dt0=DT0,
    saveat=[T1],
)


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


def plot_results(
    model: BaseAnalogCkt, data: jax.Array, switches: jax.Array, n_vars: int
):
    """
    Plot the results of the model.
    """
    ti = TimeInfo(
        t0=0.0,
        t1=T1,
        dt0=0.01,
        saveat=jnp.arange(0, T1, 0.01),
    )

    n_var_oscs = n_vars * 2
    n_plot = data.shape[0]
    figs, axes = plt.subplots(nrows=n_plot, ncols=1, figsize=(6, 2 * n_plot))
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        # axes[i].set_title(f"Oscillator {i}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Phase")
        model_out = model(ti, data[i], switches[i], 0, 0).T
        for osc_id, trace in enumerate(model_out[:n_var_oscs]):
            sign = "-" if osc_id % 2 == 0 else "+"
            label = f"{sign}x{osc_id // 2 + 1}"
            axes[i].plot(ti.saveat, trace, label=label)

        # Use 3 colores to represent the [FALSE, TRUE, BLUE] phases
        # FALSE -> red, TRUE -> green, BLUE -> blue

        for phase, name, line_style in zip(
            [BLUE_PHASE, TRUE_PHASE, FALSE_PHASE],
            ["BLUE", "TRUE", "FALSE"],
            [
                "-",
                "-.",
                "--",
            ],
        ):
            axes[i].axhline(
                y=phase,
                color="black",
                linestyle=line_style,
                label=f"{name} phase",
            )
        # Put the legend on the right side outside the plot
        axes[i].legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(SEED)

    if TASK == "3sat7clauses":
        graph, nw = create_3sat_graph(
            n_vars=3, n_clauses=7, trainable_mgr=trainable_mgr
        )
        sat_probs, sat_solutions = sat_3var7clauses_data()
        loss_fn = partial(loss_w_sol, time_info=time_info)

    elif TASK == "from_cnf":
        sat_probs = sat_from_cnf_dir(dir_path=CNF_DIR)
        prob = sat_probs[0]
        n_vars = max(abs(var) for clause in prob for var in clause)
        n_clauses = len(prob)
        graph, nw = create_3sat_graph(
            n_vars=n_vars, n_clauses=n_clauses, trainable_mgr=trainable_mgr
        )
        sat_solutions = None

    # flatten the var_oscs
    n_vars = len(nw.var_oscs)
    nodes_flat = flatten_nw_stateful_oscillators(nw)
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

    dataloader = SATDataloader(BZ, sat_probs, nw, sat_solutions)
    init_states, switches, sol, adj_mat = dataloader.__iter__().__next__()

    n_plot = N_PLOT
    print("Assignment solution:")
    print(sol[:n_plot])
    print("Solution in phase:")
    print(jnp.sin(jnp.where(sol[:n_plot], TRUE_PHASE, FALSE_PHASE) * jnp.pi))
    plot_results(model, init_states[:n_plot], switches[:n_plot], n_vars)

    model = train(model=model, loss_fn=loss_fn, dl=dataloader)
    plot_results(model, init_states[:n_plot], switches[:n_plot])
    plot_results(model, init_states[:n_plot], switches[:n_plot])

    loss_fn = partial(loss_w_sol, time_info=time_info)

    model = train(model=model, loss_fn=loss_fn, dl=dataloader)
    plot_results(model, init_states[:n_plot], switches[:n_plot], n_vars)
