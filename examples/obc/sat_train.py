import os
from functools import partial
from typing import Callable, Generator, Optional

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=multi_output_fusion"
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import spec_optimization as opt_spec
from ax import Client
from diffrax import Tsit5
from jaxtyping import PyTree
from sat_dataloader import (
    SATDataloader,
    sat_3var7clauses_data,
    sat_from_cnf_dir,
    sat_random_clauses,
)
from sat_loss import (
    approx_sat_loss,
    loss_w_sol,
    phase_to_approx_sat_loss,
    phase_to_energy,
    phase_to_sat_clause_rate,
    system_energy_loss,
)
from sat_parser import parser
from sat_utils import (
    BLUE_PHASE,
    FALSE_PHASE,
    TRUE_PHASE,
    create_3sat_graph,
    create_3sat_graph_v2,
    flatten_nw_stateful_oscillators,
    param_keys_v1,
    param_keys_v2,
    parameters_v1,
    parameters_v2,
)

import wandb
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr

jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update("jax_enable_x64", True)
args = parser.parse_args()

NETWORK_VERSION = args.network_version
if NETWORK_VERSION == "v1":
    create_graph = create_3sat_graph
elif NETWORK_VERSION == "v2":
    create_graph = create_3sat_graph_v2

SEED = args.seed

T1 = args.t1
DT0 = args.dt0
READOUT_MULTI_STEPS = args.readout_multi_steps
INITIAL_STATE = args.initial_state
STOCHASTIC = args.stochastic

BZ = args.batch_size
STEPS = args.steps
LR = args.lr

LOSS_FN = args.loss_fn

TASK = args.task
CNF_DIR: Optional[str] = args.cnf_dir
N_VARS, N_CLAUSES = args.n_vars, args.n_clauses

LOAD_PATH = args.load_path
LOAD_AX_RUN = args.load_ax_run
SAVE_PATH = args.save_path

USE_WANDB = args.wandb
RUN_NAME = args.run_name
TAG = args.tag

if USE_WANDB:
    wandb.init(
        config=vars(args),
        project="obc-sat",
        tags=[TAG] if TAG else None,
        name=RUN_NAME if RUN_NAME else None,
    )

N_PLOT = args.n_plots

AX_OPT = args.ax_opt

trainable_mgr = TrainableMgr()
optim = optax.adam(learning_rate=LR)
saveat = (
    [T1]
    if not READOUT_MULTI_STEPS
    else jnp.array([i for i in range(0, int(T1 + 1), 2)])
)
time_info = TimeInfo(
    t0=0.0,
    t1=T1,
    dt0=DT0,
    saveat=saveat,
)


@eqx.filter_jit
def make_step(model: BaseAnalogCkt, opt_state: PyTree, loss_fn: Callable, data):
    (train_loss, phase_raw), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, *data
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, train_loss, phase_raw


def make_step_ax(model: BaseAnalogCkt, params: jax.Array, loss_fn: Callable, data):
    model = eqx.tree_at(lambda m: m.a_trainable, model, params)
    (train_loss, phase_raw) = loss_fn(model, *data)
    return model, train_loss, phase_raw


def visualize_energy_and_clause_sat_rate(
    energy: jax.Array,
    clause_rate: jax.Array,
    clause_rate_approx: jax.Array,
    loss: float,
):
    """
    Visualize the energy and clause SAT rate as
        1. scatter plot of energy vs. clause SAT rate.
        2. scatter plot of approximate SAT loss vs. clause SAT rate.
        2. histogram of energy values.
        3. histogram of clause SAT rate values in [0, 1].
        4. histogram of approximated SAT loss.

    Args:
        energy (jax.Array): Energy values.
        clause_rate (jax.Array): Clause SAT rate values.
        loss (float): Loss value to be displayed in the title.
        title_prefix (str): Prefix for the plot title.
    """
    # scatter_fig, ax = plt.subplots()
    # ax.scatter(energy, clause_rate)
    # ax.set_xlabel("Energy")
    # ax.set_ylabel("Clause SAT Rate")
    # ax.set_title(
    #     f"Energy ({jnp.mean(energy):.2e}) vs. Clause SAT Rate ({jnp.mean(clause_rate):.2f}). Loss: {loss:.4f}"
    # )
    # ax.grid(True)
    # plt.tight_layout()

    # scatter_fig_approx, ax = plt.subplots()
    # ax.scatter(clause_rate_approx, clause_rate)
    # ax.set_xlabel("Approximate SAT Loss")
    # ax.set_ylabel("Clause SAT Rate")
    # ax.set_title(
    #     f"Approximate SAT Loss vs. Clause SAT Rate. Mean Loss: {jnp.mean(clause_rate_approx):.2f}. "
    #     f"Mean Clause SAT Rate: {jnp.mean(clause_rate):.2f}"
    # )
    # ax.grid(True)
    # plt.tight_layout()

    # hist_energy, ax = plt.subplots()
    # ax.hist(energy, bins=30)
    # ax.set_xlabel("Energy")
    # ax.set_ylabel("Frequency")
    # ax.set_title(
    #     f"Energy Histogram. Mean: {jnp.mean(energy):.2e}. Median: {jnp.median(energy):.2e}"
    # )
    # plt.tight_layout()

    hist_clause_rate, ax = plt.subplots()
    ax.hist(clause_rate, bins=30)
    ax.set_xlabel("Clause SAT Rate")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Clause SAT Rate Histogram. Mean: {jnp.mean(clause_rate):.2f}. Median: {jnp.median(clause_rate):.2f}"
    )
    plt.tight_layout()

    # hist_approx_sat_loss, ax = plt.subplots()
    # ax.hist(clause_rate_approx, bins=30)
    # ax.set_xlabel("Approximate SAT Loss")
    # ax.set_ylabel("Frequency")
    # ax.set_title(
    #     f"Approximate SAT Loss Histogram. Mean: {jnp.mean(clause_rate_approx):.2f}. "
    #     f"Median: {jnp.median(clause_rate_approx):.2f}"
    # )
    # plt.tight_layout()

    return [
        # scatter_fig,
        # scatter_fig_approx,
        # hist_energy,
        hist_clause_rate,
        # hist_approx_sat_loss,
    ]


def profile_nw_performance(
    model: BaseAnalogCkt,
    dl: Generator,
    loss_fn: Callable,
):
    """Profile the performance of the network by running a 8 step and visualize the energy and sat rate."""

    energy_list, clause_rate_list, loss_list = [], [], []
    approx_sat_rate_list = []
    for step, data in zip(range(8), dl):
        loss, phase_raw = loss_fn(model, *data)
        adj_mats, n_vars, probs, transform_mats = data[3:7]
        # energy = phase_to_energy(phase_raw, adj_mats)
        clause_rate = phase_to_sat_clause_rate(n_vars, phase_raw, probs)
        # approx_sat_rate = phase_to_approx_sat_loss(n_vars, phase_raw, transform_mats)
        # loss_list.append(loss)
        # energy_list.append(energy)
        clause_rate_list.append(clause_rate)
        # approx_sat_rate_list.append(approx_sat_rate)

    loss = jnp.array(loss_list).flatten()
    energy = jnp.array(energy_list).flatten()
    clause_rate = jnp.array(clause_rate_list).flatten()
    approx_sat_rate = jnp.array(approx_sat_rate_list).flatten()
    figs = visualize_energy_and_clause_sat_rate(
        energy, clause_rate, approx_sat_rate, jnp.mean(loss)
    )
    return figs


def train(model: BaseAnalogCkt, loss_fn: Callable, dl: Generator):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    best_loss = float("inf")
    best_sat_rate = 0.0
    fig_titles = [
        "Energy vs Clause SAT Rate",
        "Approximate SAT Loss vs Clause SAT Rate",
        "Energy Histogram",
        "Clause SAT Rate Histogram",
        "Approximate SAT Loss Histogram",
    ]

    if AX_OPT:
        if NETWORK_VERSION == "v1":
            param_keys = param_keys_v1
            parameters = parameters_v1
        elif NETWORK_VERSION == "v2":
            param_keys = param_keys_v2
            parameters = parameters_v2
        metric_name = "sat_rate"
        objective = f"{metric_name}"
        if LOAD_AX_RUN:
            client = Client.load_from_json_file(LOAD_AX_RUN)
            # Intialize the model with the best known parameters
            prev_best_param, _, _, _ = client.get_best_parameterization(
                use_model_predictions=False
            )
            param_flatten = jnp.array([prev_best_param[key] for key in param_keys])
            model = eqx.tree_at(
                lambda m: m.a_trainable, model, param_flatten
            )  # Update the model
        else:
            client = Client(random_seed=np.random.randint(0, 2**31))
            client.configure_experiment(parameters=parameters)
            client.configure_optimization(objective=objective)

    print("Initial trainable params:")
    print(model.a_trainable)

    for step, data in zip(range(STEPS), dl):

        if step == 0:
            # Visualize the initial energy and clause SAT rate
            figs = profile_nw_performance(
                model=model,
                dl=dl,
                loss_fn=loss_fn,
            )

            if USE_WANDB:
                for fig, title in zip(figs, fig_titles):
                    wandb.log({title: wandb.Image(fig)})
                    plt.close(fig)
            else:
                for fig in figs:
                    plt.show()
                    plt.close(fig)

        if AX_OPT:
            # Use Ax
            if step == 0 and not LOAD_AX_RUN:
                # Attach initial point with the initial trainable params
                initial_points = {
                    key: model.a_trainable[i].item() for i, key in enumerate(param_keys)
                }
                trial_index = client.attach_trial(parameters=initial_points)
                parameters = initial_points
            else:
                trial_index, parameters = list(
                    client.get_next_trials(max_trials=1).items()
                )[0]
            param_flatten = jnp.array([parameters[key] for key in param_keys])
            model, train_loss, phase_raw = make_step_ax(
                model, param_flatten, loss_fn, data
            )
            sat_rate = phase_to_sat_clause_rate(data[4], phase_raw, data[5])
            opt_data = {
                metric_name: sat_rate.mean().item(),
            }
            client.complete_trial(
                trial_index=trial_index,
                raw_data=opt_data,
            )

        else:
            # Use gradient descent
            model, opt_state, train_loss, phase_raw = make_step(
                model, opt_state, loss_fn, data
            )
        sat_rate = jnp.mean(phase_to_sat_clause_rate(data[4], phase_raw, data[5]))

        print(
            f"\nStep {step}, Train loss: {train_loss}, Clause SAT Rate: {jnp.mean(sat_rate)}"
        )
        print("Trainable params")
        print(model.a_trainable)

        if USE_WANDB:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "sat_rate": sat_rate,
                    "step": step,
                }
            )

        if sat_rate > best_sat_rate or (
            sat_rate == best_sat_rate and train_loss < best_loss
        ):
            best_sat_rate = sat_rate
            best_loss = train_loss
            if SAVE_PATH:
                eqx.tree_serialise_leaves(SAVE_PATH, model)
            eqx.tree_serialise_leaves("tmp.eqx", model)  # Temporary save loading later

    best_model = eqx.tree_deserialise_leaves("tmp.eqx", model)
    figs = profile_nw_performance(
        model=best_model,
        dl=dl,
        loss_fn=loss_fn,
    )
    if USE_WANDB:
        for fig, title in zip(figs, fig_titles):
            wandb.log({title: wandb.Image(fig)})
            plt.close(fig)
    else:
        for fig in figs:
            plt.show()
            plt.close(fig)

    if AX_OPT and SAVE_PATH:
        client.save_to_json_file(SAVE_PATH)

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

    if LOSS_FN == "approx_sat":
        loss_fn_base = approx_sat_loss
    elif LOSS_FN == "energy":
        loss_fn_base = system_energy_loss

    if TASK == "3var7clauses":
        graph, nw = create_graph(n_vars=3, n_clauses=7, trainable_mgr=trainable_mgr)
        sat_probs, sat_solutions = sat_3var7clauses_data()
        loss_fn = partial(loss_w_sol, time_info=time_info)

    elif TASK == "from_cnf":
        sat_probs = sat_from_cnf_dir(dir_path=CNF_DIR)
        prob = sat_probs[0]
        n_vars = max(abs(var) for clause in prob for var in clause)
        n_clauses = len(prob)
        graph, nw = create_graph(
            n_vars=n_vars, n_clauses=n_clauses, trainable_mgr=trainable_mgr
        )
        sat_solutions = None
        loss_fn = partial(loss_fn_base, time_info=time_info)

    elif TASK == "random":
        assert (
            N_VARS is not None and N_CLAUSES is not None
        ), "For 'random' task, --n_vars and --n_clauses must be specified."
        sat_probs = sat_random_clauses(
            n_vars=N_VARS, n_clauses=N_CLAUSES, n_prob=BZ * 1024
        )
        n_vars, n_clauses = N_VARS, N_CLAUSES
        graph, nw = create_graph(
            n_vars=n_vars, n_clauses=n_clauses, trainable_mgr=trainable_mgr
        )
        sat_solutions = None
        loss_fn = partial(loss_fn_base, time_info=time_info)

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
        is_stochastic=STOCHASTIC,
        solver=Tsit5(),
    )
    if LOAD_PATH:
        model = eqx.tree_deserialise_leaves(LOAD_PATH, model)
    nw.set_var_clause_cpls_args_idx(model=model)

    dataloader = SATDataloader(INITIAL_STATE, BZ, sat_probs, nw, sat_solutions)
    init_states, switches, sol, adj_mat, n_vars, probs, transform_mats, seeds = next(
        dataloader.__iter__()
    )

    # n_plot = N_PLOT
    # plot_results(model, init_states[:n_plot], switches[:n_plot], n_vars)

    model = train(model=model, loss_fn=loss_fn, dl=dataloader)
    # plot_results(model, init_states[:n_plot], switches[:n_plot])
