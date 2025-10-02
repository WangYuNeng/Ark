import os
from typing import Generator, Optional

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=multi_output_fusion"
import argparse

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import matplotlib.pyplot as plt
import numpy as np
from ax import Client, RangeParameterConfig
from sat_dataloader import sat_from_cnf_dir
from sat_utils import Problem, n_sat_clauses

import wandb
from ark.optimization.base_module import TimeInfo

jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update("jax_enable_x64", True)

FALSE_PHASE, TRUE_PHASE = 0, jnp.pi


def locking_2x(x, lock_strength: float, alpha, t: float) -> jax.Array:
    # clock that goes from -0.1 to 0.9
    anneal_cycle = jnp.tanh(alpha * jnp.cos(jnp.pi * t))
    lock_strength = lock_strength * (anneal_cycle / 2 + 0.4)
    return lock_strength * jnp.sin(2 * x)


def kuramoto(
    t: jax.typing.DTypeLike, y: jax.Array, args: tuple[jax.Array, jax.Array]
) -> jax.Array:
    """Kuramoto model ODE function"""
    k_arr, J_matrix = args
    kc, ks, _, alpha = k_arr
    phi_diff = y[:, None] - y[None, :]  # shape (N, N)
    coupling_contrib = J_matrix * jnp.sin(phi_diff)
    coupling_term: jax.Array = kc * jnp.sum(coupling_contrib, axis=1)
    locking_term = locking_2x(y, ks, alpha, t)
    # Paper use dydt = - coupling_term - locking_term
    # But that perform poorly. Maybe there is a sign I misread.
    dydt = coupling_term - locking_term
    # Mask the auxiliary variable (last oscillator) to have zero dynamics
    dydt = dydt.at[-1].set(0)
    return dydt


def white_noise(t, y, args):
    """White noise function"""
    kn = args[0][2]
    dydt = kn * jnp.ones_like(y)
    # Mask the auxiliary variable (last oscillator) to have zero noise
    dydt = dydt.at[-1].set(0)
    return dydt


class OscNetwork(eqx.Module):
    """Implementation of the 3OR oscillator network from
    https://link.springer.com/chapter/10.1007/978-3-031-63742-1_19
    """

    params: jax.Array  # (ks, kc, kn, alpha)

    def __init__(self) -> None:
        self.params = jnp.ones(4)

    def __call__(
        self,
        initial_state: jax.Array,
        J_matrix: jax.Array,
        noise_seed: jax.typing.DTypeLike,
        time_info: TimeInfo,
        max_steps: int = 4096 * 16,
    ) -> jax.Array:
        args = (self.params, J_matrix)
        initial_state = initial_state.at[-1].set(jnp.pi)  # Set aux to TRUE
        ode_term = diffrax.ODETerm(kuramoto)
        brownian = diffrax.VirtualBrownianTree(
            time_info.t0,
            time_info.t1,
            tol=time_info.dt0 / 2,
            shape=initial_state.shape,
            key=jax.random.PRNGKey(noise_seed),
        )
        vector_field = lambda t, y, args: lineax.DiagonalLinearOperator(
            white_noise(t, y, args)
        )
        brownian_term = diffrax.ControlTerm(vector_field, brownian)
        solution = diffrax.diffeqsolve(
            terms=diffrax.MultiTerm(ode_term, brownian_term),
            solver=diffrax.Tsit5(),
            t0=time_info.t0,
            t1=time_info.t1,
            dt0=time_info.dt0,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=time_info.saveat),
            stepsize_controller=diffrax.ConstantStepSize(),
            args=args,
            max_steps=max_steps,
        )

        return solution.ys


def gen_j_matrix(prob: Problem, n_vars: int) -> jax.Array:
    """Generate the J matrix for the given SAT problem.

    Args:
        prob: A SAT problem represented as a list of clauses.
        n_vars: The number of variables in the SAT problem.
    Returns:
        A J matrix of shape (n_vars + n_clause + 1, n_vars + n_clause + 1).
        The last row and column correspond to the auxiliary variable.
    """

    n_clauses = len(prob)
    n_oscillators = n_vars + n_clauses + 1  # +1 for the auxiliary variable
    j_mat = np.zeros((n_oscillators, n_oscillators))

    def connect(id0: int, id1: int, val: float):
        j_mat[id0, id1] += val
        j_mat[id1, id0] += val

    for clause_idx, clause in enumerate(prob):
        clause_osc_idx = n_vars + clause_idx
        aux_idx = n_oscillators - 1
        a, b, c = clause.to_list()
        a_idx, b_idx, c_idx = abs(a) - 1, abs(b) - 1, abs(c) - 1
        sign_a, sign_b, sign_c = np.sign(clause.to_list())
        a_aux = sign_a
        a_b = sign_a * sign_b
        b_aux = sign_b
        c_aux = -sign_c
        a_clause = -2 * sign_a
        b_clause = -2 * sign_b
        c_clause = sign_c
        clause_aux = -3
        connect(a_idx, aux_idx, a_aux)
        connect(a_idx, b_idx, a_b)
        connect(b_idx, aux_idx, b_aux)
        connect(c_idx, aux_idx, c_aux)
        connect(a_idx, clause_osc_idx, a_clause)
        connect(b_idx, clause_osc_idx, b_clause)
        connect(c_idx, clause_osc_idx, c_clause)
        connect(clause_osc_idx, aux_idx, clause_aux)

    return jnp.array(j_mat)


class SATDataloader:
    """
    A dataloader to prepare the SAT problem for the OBC.

    Args:
        sat_probs: A list of SAT problems, each represented as a list of clauses.
        batch_size: The batch size for the dataloader.
        osc_network: The OBC network to be used for the SAT problem, must have the same # of variables and
            clauses as the SAT problem.
    """

    initial_state: str
    batch_size: int
    sat_probs: list[Problem]
    n_vars: int

    def __init__(
        self,
        batch_size: int,
        sat_probs: list[Problem],
        n_vars: int,
    ):
        self.batch_size = batch_size
        self.sat_probs = sat_probs
        self.n_vars = n_vars

    def __iter__(
        self,
    ) -> Generator[tuple[jax.Array, jax.Array, jax.Array, jax.Array, int], None, None]:

        batch_size = self.batch_size
        n_oscillators = (
            self.n_vars + len(self.sat_probs[0]) + 1
        )  # +1 for the auxiliary variable

        while True:
            initial_states = np.random.rand(batch_size, n_oscillators) * 2 * np.pi
            sampled_prob_idx = np.random.choice(
                len(self.sat_probs), batch_size, replace=True
            )
            probs = [
                self.sat_probs[prob_idx].to_list() for prob_idx in sampled_prob_idx
            ]
            J_matrices = [
                gen_j_matrix(self.sat_probs[prob_idx], self.n_vars)
                for prob_idx in sampled_prob_idx
            ]
            noise_seed = np.random.randint(0, 2**31 - 1, size=(batch_size,))
            yield (
                jnp.array(initial_states),
                jnp.array(probs),
                jnp.array(J_matrices),
                jnp.array(noise_seed),
                self.n_vars,
            )


def sat_clause_rate_score(
    model: OscNetwork,
    init_states: jax.Array,
    problems: jax.Array,
    J_matrices: jax.Array,
    noise_seed: jax.Array,
    n_vars: int,
    time_info: TimeInfo,
) -> jax.Array:
    y_raw = jax.vmap(model, in_axes=(0, 0, 0, None))(
        init_states, J_matrices, noise_seed, time_info
    )
    assignment_phase = y_raw[:, :, :n_vars]
    # Map the variable phases to boolean assignments
    modular_phase = jnp.mod(assignment_phase, 2.0 * jnp.pi)
    threshold = (TRUE_PHASE - FALSE_PHASE) / 2
    bool_assignments = jnp.array(
        [jnp.abs(phase - TRUE_PHASE) < threshold for phase in modular_phase]
    )

    # Calculate the number of satisfied clauses for each problem
    n_sat_clause_list = []
    for clauses, assignments_in_run in zip(problems, bool_assignments):
        # Count the number of satisfied clauses for each run
        # Record the assignment that satisfies the most clauses
        best_n_sat_clause = 0
        for assignment in assignments_in_run:
            best_n_sat_clause = max(
                best_n_sat_clause, n_sat_clauses(clauses, assignment)
            )
        n_sat_clause_list.append(best_n_sat_clause)

    # Convert the list to a jax array
    n_satisfied_clauses = jnp.array(n_sat_clause_list)

    # Calculate the ratio of satisfied clauses
    n_clauses = jnp.array([len(prob) for prob in problems])
    ratio_satisfied = n_satisfied_clauses / n_clauses
    return ratio_satisfied


def profile_nw_performance(
    model: OscNetwork,
    dl: Generator,
    time_info: TimeInfo,
):
    """Profile the performance of the network by running a 8 step and visualize the energy and sat rate."""

    clause_rate_list = []
    for step, data in zip(range(8), dl):
        sat_rate = sat_clause_rate_score(model, *data, time_info=time_info)
        clause_rate_list.append(sat_rate)

    clause_rate = jnp.array(clause_rate_list).flatten()
    hist_clause_rate, ax = plt.subplots()
    ax.hist(clause_rate, bins=30)
    ax.set_xlabel("Clause SAT Rate")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Clause SAT Rate Histogram. Mean: {jnp.mean(clause_rate):.2f}. Median: {jnp.median(clause_rate):.2f}"
    )
    plt.tight_layout()
    return [hist_clause_rate]


def train(
    model: OscNetwork,
    dl: Generator,
    STEPS: int,
    time_info: TimeInfo,
    loss_fn: str,
    SAVE_PATH: Optional[str] = None,
    LOAD_AX_RUN: Optional[str] = None,
    USE_WANDB: bool = False,
):
    best_sat_rate = 0.0
    fig_titles = [
        "Energy vs Clause SAT Rate",
        "Approximate SAT Loss vs Clause SAT Rate",
        "Energy Histogram",
        "Clause SAT Rate Histogram",
        "Approximate SAT Loss Histogram",
    ]

    param_keys = ["ks", "kc", "kn", "alpha"]
    parameters = [
        RangeParameterConfig(name="ks", parameter_type="float", bounds=[0.001, 10.0]),
        RangeParameterConfig(name="kc", parameter_type="float", bounds=[0.001, 10.0]),
        RangeParameterConfig(name="kn", parameter_type="float", bounds=[0.001, 10.0]),
        RangeParameterConfig(name="alpha", parameter_type="float", bounds=[0.0, 5.0]),
    ]
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
    print(model.params)

    for step, data in zip(range(STEPS), dl):

        if step == 0:
            # Visualize the initial energy and clause SAT rate
            figs = profile_nw_performance(
                model=model,
                dl=dl,
                time_info=time_info,
            )

            if USE_WANDB:
                for fig, title in zip(figs, fig_titles):
                    wandb.log({title: wandb.Image(fig)})
                    plt.close(fig)
            else:
                for fig in figs:
                    plt.show()
                    plt.close(fig)

        # Use Ax
        if step == 0 and not LOAD_AX_RUN:
            # Attach initial point with the initial trainable params
            initial_points = {
                key: model.params[i].item() for i, key in enumerate(param_keys)
            }
            trial_index = client.attach_trial(parameters=initial_points)
            parameters = initial_points
        else:
            trial_index, parameters = list(
                client.get_next_trials(max_trials=1).items()
            )[0]
        param_flatten = jnp.array([parameters[key] for key in param_keys])
        model = eqx.tree_at(
            lambda m: m.params, model, param_flatten
        )  # Update the model
        sat_rate = (
            sat_clause_rate_score(model, *data, time_info=time_info).mean().item()
        )

        opt_data = {
            metric_name: sat_rate,
        }
        client.complete_trial(
            trial_index=trial_index,
            raw_data=opt_data,
        )

        print(f"\nStep {step}, Clause SAT Rate: {sat_rate}")
        print("Trainable params")
        print(model.params)

        if USE_WANDB:
            wandb.log(
                {
                    "sat_rate": sat_rate,
                    "step": step,
                }
            )

        if sat_rate > best_sat_rate:
            best_sat_rate = sat_rate
            if SAVE_PATH:
                eqx.tree_serialise_leaves(SAVE_PATH, model)
            eqx.tree_serialise_leaves("zzz.eqx", model)  # Temporary save loading later

    best_model = eqx.tree_deserialise_leaves("zzz.eqx", model)
    figs = profile_nw_performance(
        model=best_model,
        dl=dl,
        time_info=time_info,
    )
    if USE_WANDB:
        for fig, title in zip(figs, fig_titles):
            wandb.log({title: wandb.Image(fig)})
            plt.close(fig)
    else:
        for fig in figs:
            plt.show()
            plt.close(fig)

    if SAVE_PATH:
        client.save_to_json_file(SAVE_PATH)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--cnf_dir", type=str, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=428,
        help="Random seed.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=40.0,
        help="The time duration for the simulation.",
    )
    parser.add_argument(
        "--dt0",
        type=float,
        default=0.01,
        help="The time step size for the simulation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the best model and ax optimization state.",
    )
    parser.add_argument(
        "--load_ax_run",
        type=str,
        default=None,
        help="Path to load the ax optimization state.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb to log the training process.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the wandb run.",
    )

    args = parser.parse_args()

    T1, DT0 = args.t1, args.dt0
    SEED = args.seed

    STEPS = args.steps
    BATCH_SIZE = args.batch_size
    SAVE_PATH = args.save_path
    LOAD_AX_RUN = args.load_ax_run
    USE_WANDB = args.wandb
    TAG = args.tag
    RUN_NAME = args.run_name

    if USE_WANDB:
        wandb.init(
            config=vars(args),
            project="obc-sat",
            tags=[TAG] if TAG else None,
            name=RUN_NAME if RUN_NAME else None,
        )

    saveat = jnp.array([i for i in range(0, int(T1 + 1), 2)])
    time_info = TimeInfo(
        t0=0.0,
        t1=T1,
        dt0=DT0,
        saveat=saveat,
    )

    sat_probs = sat_from_cnf_dir(dir_path=args.cnf_dir)
    prob = sat_probs[0]
    n_vars = max(abs(var) for clause in prob for var in clause)
    n_clauses = len(prob)

    dataloader = SATDataloader(
        batch_size=BATCH_SIZE, sat_probs=sat_probs, n_vars=n_vars
    )

    model = OscNetwork()
    train(
        model=model,
        dl=dataloader,
        STEPS=STEPS,
        time_info=time_info,
        loss_fn="sat_rate",
        SAVE_PATH=SAVE_PATH,
        LOAD_AX_RUN=LOAD_AX_RUN,
        USE_WANDB=USE_WANDB,
    )
