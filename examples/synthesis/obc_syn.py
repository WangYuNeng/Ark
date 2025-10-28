# Demonstrate syntheizing function with oscillators

from itertools import product
from typing import Callable

import cvxpy as cp
import diffrax
import jax
import jax.numpy as jnp
import lineax
import matplotlib.pyplot as plt
import numpy as np
from cvc5.pythonic import Const, Int, Ints, Or, Real, Reals, RealVal, Solver, sat
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def bool_to_phase(b):
    if isinstance(b, (bool, int)):
        return 1 if b else -1
    elif isinstance(b, list):
        return [1 if bi else -1 for bi in b]
    elif isinstance(b, np.ndarray):
        return np.where(b, 1, -1)
    else:
        raise ValueError("Input must be bool or np.ndarray of bools")


def phase_to_bool(p):
    if isinstance(p, (int, float)):
        return 1 if p == 1 else 0
    elif isinstance(p, list):
        return [1 if pi == 1 else 0 for pi in p]
    elif isinstance(p, np.ndarray):
        return np.where(p == 1, 1, 0)
    else:
        raise ValueError("Input must be int, float, list, or np.ndarray")


def fit_temperature(
    phase_to_energy: dict[tuple, int], phase_to_measurements: dict[tuple, int]
):
    """Fit an temperature parameter from calculated energy and measured distribution.

    Args:
        phase_to_energy (dict[tuple, int]): Mapping from state (tuple of phases) to energy value
        phase_to_measurements (dict[tuple, int]): Mapping from state (tuple of phases) to count of occurrences
    Returns:
        float: Fitted temperature parameter beta
        float: KL divergence value
        dict[tuple, int]: Fitted distribution over states
    """

    assert set(phase_to_energy.keys()) == set(
        phase_to_measurements.keys()
    ), "Energy and measurement keys must match"
    n_measurement = sum(phase_to_measurements.values())
    phase_to_probs = {k: v / n_measurement for k, v in phase_to_measurements.items()}

    beta = cp.Variable(1)
    kl_div = 0

    # Minimize KL divergence between the true distribution (prob) and the fitting distribution
    log_Q_denominator = cp.log_sum_exp(
        cp.vstack([-beta * e for e in phase_to_energy.values()])
    )
    for phases in phase_to_energy.keys():
        energy = phase_to_energy[phases]
        prob = phase_to_probs[phases]
        log_Q_numerator = -beta * energy
        if prob != 0:
            kl_div += (
                prob
                * (cp.log(prob) - (log_Q_numerator - log_Q_denominator))
                / np.log(2)
            )
    prob = cp.Problem(cp.Minimize(kl_div))
    prob.solve()
    fitted_beta = beta.value[0].item()
    fitted_distribution = {
        k: np.exp(-fitted_beta * e).item() for k, e in phase_to_energy.items()
    }
    partition = sum(fitted_distribution.values())
    fitted_distribution = {
        k: v / partition * n_measurement for k, v in fitted_distribution.items()
    }

    return fitted_beta, prob.value, fitted_distribution


def random_simulation(
    J: np.ndarray,
    n_sim=256,
    Kc=1,
    Kl=1,
    Kt=0.1,
    Kt_ratio=100,
    super_harmonic=2,
    t_end=2,
    dt0=0.01,
    anneal=False,
    plot=False,
):
    """Simulate Kuramoto model with random initial states

    Args:
        J (np.ndarray): Coupling matrix
        n_sim (int, optional): Number of simulation runs. Defaults to 256.
        Kc (int, optional): Coupling scale factor. Defaults to 1.
        Kl (int, optional): Injection locking scale factor. Defaults to 1.
        super_harmonic (int, optional): Injectio locking frequency. Defaults to 2.
        anneal (bool, optional): Whether to use an annealing schedule for locking. Defaults to False.

    Returns:
        np.ndarray: (n_sim, J.shape[0] - 1) the final state of each simulation run
    """
    n_free_osc = J.shape[0] - 1
    J = jnp.array(J)

    def obc_ode(t, y, args):
        y_w_fixed = jnp.concatenate([y, jnp.array([jnp.pi])])  # Fixed oscillator at pi
        y_diff = y_w_fixed[:, None] - y_w_fixed[None, :]
        coupling = Kc * jnp.sum(J * jnp.sin(y_diff), axis=1)
        locking = Kl * jnp.sin(super_harmonic * y_w_fixed)
        dydt = (coupling - locking)[:-1]  # Exclude the fixed oscillator
        return dydt

    def noise_ode(t, y, args):
        if not anneal:
            Kt_anneal = Kt
        else:
            # From Kt_ratio * Kt to Kt exponentially over [0, t_end]
            Kt_anneal = Kt_ratio * Kt * jnp.exp(-jnp.log(Kt_ratio) * t / t_end)
        return lineax.DiagonalLinearOperator(Kt_anneal * jnp.ones(n_free_osc))

    ode_term = diffrax.ODETerm(obc_ode)
    ts = jnp.arange(0, t_end, dt0)

    @jax.jit
    def sim(y0: jax.Array, seed: int):
        brownian = diffrax.VirtualBrownianTree(
            t0=0,
            t1=t_end,
            tol=dt0 / 2,
            shape=(n_free_osc,),
            key=jax.random.PRNGKey(seed),
        )

        brownian_term = diffrax.ControlTerm(noise_ode, brownian)
        solution = diffrax.diffeqsolve(
            terms=diffrax.MultiTerm(ode_term, brownian_term),
            solver=diffrax.Euler(),
            t0=0,
            t1=t_end,
            dt0=dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=int(t_end // dt0 * 2),
        )

        return solution.ys.T

    y0s = jnp.array(np.random.uniform(-np.pi, np.pi, (n_sim, n_free_osc)))
    seed = jnp.array(np.random.randint(0, 2**32 - 1, n_sim))
    results = jax.vmap(sim, in_axes=(0, 0))(y0s, seed)

    # Plot
    if plot:
        phases = results[0]
        plt.figure(figsize=(10, 5))
        for i in range(phases.shape[0]):
            plt.plot(ts, phases[i], label=f"oscillator {i}")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    return jnp.array(results), ts


def rectify_phases(phases: np.ndarray, super_harmonic: int):
    """Rectify oscillator phases to discrete states based on super harmonic injection locking.

    Args:
        phases (np.ndarray): Array of oscillator phases
        super_harmonic (int): Injection locking frequency

    Returns:
        np.ndarray: Rectified phases
    """
    rect_vals = jnp.array(
        [(2 * jnp.pi / super_harmonic) * i for i in range(super_harmonic + 1)]
    )
    rectified_phases = jnp.array(
        [rect_vals[jnp.argmin(jnp.abs(rect_vals - (v % (2 * jnp.pi))))] for v in phases]
    )
    # Map 2pi back to 0
    rectified_phases = jnp.where(rectified_phases == 2 * jnp.pi, 0, rectified_phases)
    return rectified_phases


def energy_fn(J_mat: np.ndarray, v: np.ndarray, exclude_ref_energy: bool = False):
    """Calculate the energy of a state given coupling matrix and oscillator phases.

    Args:
        J_mat (np.ndarray): Coupling matrix
        v (np.ndarray): Oscillator phases, values should be -1 or 1
        exclude_ref_energy (bool, optional): Whether to exclude the reference oscillator from energy calculation.
            Defaults to False.
    Returns:
        Same type as elements in the ndarray: Energy value
    """
    n_var = len(v)
    assert J_mat.shape == (n_var, n_var)
    assert (
        v[-1] == 1
    ), "Last oscillator must be the fixed reference oscillator with phase 1"

    # Pairwise products
    v_diff = v[:, None] * v[None, :]

    # Zero diagonal
    v_diff = v_diff * (1 - np.eye(n_var))

    energy_per_coupling = J_mat * v_diff

    # The couplings to fixed oscillator does not contribute to energy
    if exclude_ref_energy:
        energy_per_coupling[-1, :] = 0
    return energy_per_coupling.sum()


def KL_divergence_to_ideal(
    phase_to_energy: dict[tuple, int],
    phase_to_measurements: dict[tuple, int],
    n_io_var: int,
):
    """Calculate the KL divergence between the measured distribution and the ideal distribution.

    Ideal distribution is from infinite beta, i.e., probability 1 for lowest energy states and 0
      for others. Here ideal distribution is P and measured distribution is Q for KL divergence.

    Args:
        phase_to_energy (dict[tuple, int]): Mapping from state (tuple of phases) to energy value
        phase_to_measurements (dict[tuple, int]): Mapping from state (tuple of phases) to count of occurrences
        n_io_var (int): Number of input/output variables (oscillators).
    Returns:
        float: KL divergence value
    """

    assert set(phase_to_energy.keys()) == set(
        phase_to_measurements.keys()
    ), "Energy and measurement keys must match"
    n_measurement = sum(phase_to_measurements.values())
    phase_to_probs = {k: v / n_measurement for k, v in phase_to_measurements.items()}

    min_energy = min(phase_to_energy.values())
    min_energy_states = set()
    agg_phase_to_prob = {}
    for phase, energy in phase_to_energy.items():
        io_phase = phase[:n_io_var]
        if energy == min_energy:
            min_energy_states.add(io_phase)
        agg_phase_to_prob[io_phase] = (
            agg_phase_to_prob.get(io_phase, 0) + phase_to_probs[phase]
        )

    n_ideal_states = len(min_energy_states)
    ideal_prob = 1 / n_ideal_states

    kl_div = 0
    for phase, q in agg_phase_to_prob.items():
        if phase in min_energy_states:
            p = ideal_prob
            if q == 0:
                print("Measured distribution has zero probability for an ideal state")
                return None
            kl_div += p * (np.log2(p) - np.log2(q))

    return kl_div


def synthesize_general(
    logic_fn: Callable,
    n_io_var: int,
    n_aux_osc: int,
    symmetric: bool = True,
    constrain_coupling: int = 0,
    constrain_energy_threshold: bool = False,
    exclude_ref_energy: bool = False,
):
    """Synthesize oscillator coupling matrix to implement a logic function

    Args:
        logic_fn (Callable): Logic function to implement. Takes n_io_var boolean inputs and returns
            a boolean output.
        n_io_var (int): Number of input/output variables (oscillators).
        n_aux_osc (int): Number of auxiliary oscillators (free variables).
        symmetric (bool, optional): Whether to constrain the coupling matrix to be symmetric. Defaults to True.
        constrain_coupling (int, optional): Maximum number of non-zero couplings. Defaults to 0 (no constraint).
        constrain_energy_threshold (bool, optional): Whether to constrain the energy of non-solution states to be above
            a threshold and solution states to be below a threshold. Defaults to False.
        exclude_ref_energy (bool, optional): Whether to exclude the reference oscillator from energy calculation.
            Defaults to False.
    Returns:
        model: The solver model if a solution is found, else None.
        J_mat: The coupling matrix if a solution is found, else None.
    """

    # 5 oscillators are input/output, 1 , rest are free variables
    n_osc = n_io_var + n_aux_osc + 1
    phases = [-1, 1]
    fixed_oscs = [1]  # Two fixed oscillators as references

    constraints = []

    # NxN coupling matrix
    # J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J_mat = np.array(J).reshape((n_osc, n_osc))

    # Auxiliary variable: minimum energy
    e_min = Int("e_min")
    e_thresh = Int(
        "e_thresh"
    )  # Want that total energy of non-solution states is above e_thresh and total energy of solution states is below e_min
    constraints.append(e_thresh >= e_min)

    # Enumerate all io and free boolean values
    io_val_tab = [list(v) for v in product(*[phases for _ in range(n_io_var)])]
    aux_val_tab = [list(v) for v in product(*[phases for _ in range(n_aux_osc)])]

    for ios_phase in io_val_tab:
        ios_bool = phase_to_bool(ios_phase)
        possible_solution_states = []
        non_solution_states = []
        for free_vals in aux_val_tab:
            v = np.array(ios_phase + free_vals + fixed_oscs)
            e = energy_fn(J_mat, v, exclude_ref_energy=exclude_ref_energy)
            if logic_fn(*ios_bool):
                # Solution states have energy at least e_min
                possible_solution_states.append(e)
            else:
                # Non-solution states have energy strictly greater than e_min
                constraints.append(e > e_min)
                non_solution_states.append(e)
        if possible_solution_states:
            # At least one solution state has energy equal to e_min
            constraints.append(Or(*[e == e_min for e in possible_solution_states]))
            if constrain_energy_threshold:
                constraints.append(sum(possible_solution_states) <= e_thresh)
        if non_solution_states and constrain_energy_threshold:
            constraints.append(sum(non_solution_states) > e_thresh)

    # Symmetry and zero diagonal constraints
    for i in range(n_osc):
        constraints.append(J[i * n_osc + i] == 0)
        if symmetric:
            for j in range(i + 1, n_osc):
                constraints.append(J[i * n_osc + j] == J[j * n_osc + i])

    # Constraint the number of non-zero couplings
    if constrain_coupling > 0:
        coupling_indicators = [
            Int(f"coup_ind_{i}_{j}") for i in range(n_osc) for j in range(n_osc)
        ]
        for idx, coup_ind in enumerate(coupling_indicators):
            constraints.append(
                Or(J[idx] == 0, coup_ind == 1)
            )  # If coupling non-zero, indicator is 1
            constraints.append(
                Or(J[idx] != 0, coup_ind == 0)
            )  # If coupling is zero, indicator is 0
        constraints.append(
            sum(coupling_indicators) <= constrain_coupling
        )  # Limit the number of couplings

    # Solve
    s = Solver()
    s.add(*constraints)
    if s.check() != sat:
        print("UNSAT, no solution found")
        return None, None

    else:
        m = s.model()
        return m, J_mat


def validate_synthesis(
    model,
    J_mat: np.ndarray,
    logic_fn: Callable,
    n_io_var: int,
    n_aux_osc: int,
    Kc: int = 1,
    Kl: int = 1,
    Kt: int = 0.1,
    Kt_ratio: int = 100,
    t_span=(0, 2),
    dt0: float = 0.01,
    anneal: bool = False,
    plot: bool = True,
):
    """Validate the synthesis result by calculating energies and simulating the system.

    Args:
        model: The solver model from synthesis.
        J_mat (np.ndarray): The coupling matrix from synthesis.
        logic_fn (Callable): Logic function to implement. Takes n_io_var boolean inputs and returns
            a boolean output.
        n_io_var (int): Number of input/output variables (oscillators).
        n_aux_osc (int): Number of auxiliary oscillators (free variables).
        Kc (int, optional): Coupling scale factor for simulation. Defaults to 1.
        Kl (int, optional): Injection locking scale factor for simulation. Defaults to 1.
        t_span (tuple, optional): Simulation time span. Defaults to (0, 2).
        anneal (bool, optional): Whether to use an annealing schedule for locking in simulation. Defaults to False.
        plot (bool, optional): Whether to plot the simulation result histogram. Defaults to True.
    Returns:
        dict: Histogram of final states from simulation.
        tuple: (w_ref_energy_data, wo_ref_energy_data) where each is a dict containing:
            - beta: Fitted temperature parameter
            - kl_div: KL divergence value
            - distribution: Fitted distribution over states
    """
    m = model
    n_osc = n_io_var + n_aux_osc + 1
    fixed_oscs = [1]
    phases = [-1, 1]
    io_val_tab = [list(v) for v in product(*[phases for _ in range(n_io_var)])]
    aux_val_tab = [list(v) for v in product(*[phases for _ in range(n_aux_osc)])]

    J_val = np.array(
        [[m[J_mat[i, j]].as_long() for j in range(n_osc)] for i in range(n_osc)]
    )
    # Normalize J to have mean absolute value of 1
    J_val = J_val / np.mean(np.abs(J_val))

    print("Normalized Coupling matrix:")
    print(J_val)

    # List the truth table and corresponding energies
    print("Truth table and energies:")
    titles = ["I/O", "E_w_ref", "E_wo_ref"]
    phase_to_energy_w_ref, phase_to_energy_wo_ref = {}, {}
    print(f"{titles[0]:<20}{titles[1]:<10}{titles[2]:<10}")
    for ios_phase in io_val_tab:
        es_w_ref, es_wo_ref = [], []
        for aux_vals in aux_val_tab:
            v = np.array(ios_phase + aux_vals + fixed_oscs)
            energy_w_ref = m[energy_fn(J_mat, v, exclude_ref_energy=False)].as_long()
            energy_wo_ref = m[energy_fn(J_mat, v, exclude_ref_energy=True)].as_long()
            es_w_ref.append(energy_w_ref)
            es_wo_ref.append(energy_wo_ref)
            phase_to_energy_w_ref[tuple(phase_to_bool(ios_phase + aux_vals))] = (
                energy_w_ref
            )
            phase_to_energy_wo_ref[tuple(phase_to_bool(ios_phase + aux_vals))] = (
                energy_wo_ref
            )
        ios_bool = phase_to_bool(ios_phase)
        print(f"{str(ios_bool):<20}{str(es_w_ref):<10} {str(es_wo_ref):<10}")

    # Validate with simulation
    traces, ts = random_simulation(
        J_val,
        n_sim=2 ** (n_osc + 6),
        Kl=Kl,
        Kc=Kc,
        Kt=Kt,
        Kt_ratio=Kt_ratio,
        anneal=anneal,
        t_end=t_span[1],
        dt0=dt0,
    )
    sim_results = []
    for trace in traces:
        sim_results.append(rectify_phases(trace[:, -1], super_harmonic=2))

    # Plot the sim result histogram -- count how many times each input/output combination occurs
    hist = {
        tuple(phase_to_bool(list(p))): 0
        for p in product(*[phases for _ in range(n_io_var + n_aux_osc)])
    }
    for res in sim_results:
        key = tuple([int(bool(p)) for p in res[: n_io_var + n_aux_osc]])
        hist[key] += 1

    # Rotate the x-axis labels for better readability
    # Highlight the valid states
    if plot:
        plt.bar(
            range(len(hist)), hist.values(), tick_label=[str(k) for k in hist.keys()]
        )

    w_ref_energy_data, wo_ref_energy_data = {}, {}
    for label, p2e, color, data in zip(
        ["w/ Ref energy", "w/o Ref energy"],
        [phase_to_energy_w_ref, phase_to_energy_wo_ref],
        ["orange", "gold"],  # contrast to blue bars
        [w_ref_energy_data, wo_ref_energy_data],
    ):
        beta, kl_div, distribution = fit_temperature(p2e, hist)
        data["beta"] = beta
        data["kl_div"] = kl_div
        data["distribution"] = distribution

        # Plot the fitted distribution as a line on top of the histogram
        if plot:
            plt.plot(
                range(len(hist)),
                [distribution[k] for k in hist.keys()],
                label=f"{label}, beta={beta:.2f}, KL Div={kl_div:.4f}",
                marker="o",
                color=color,
            )

    kl_div_to_ideal = KL_divergence_to_ideal(
        phase_to_energy_w_ref, hist, n_io_var=n_io_var
    )
    # Highlight the valid states
    if plot:
        valid_states = [
            tuple(phase_to_bool(ios_phase))
            for ios_phase in io_val_tab
            if logic_fn(*phase_to_bool(ios_phase))
        ]
        for i, k in enumerate(hist.keys()):
            ios = k[:n_io_var]
            if ios in valid_states:
                plt.gca().get_xticklabels()[i].set_color("red")
        plt.xticks(rotation=90)
        plt.xlabel(f"{n_io_var} Input/Output States")
        plt.ylabel("Count")
        if kl_div_to_ideal is None:
            plt.title("Final State Distribution")
        else:
            plt.title(
                f"Final State Distribution, KL Div to Ideal={kl_div_to_ideal:.4f}"
            )
        plt.tight_layout()
        plt.legend()
        plt.show()

    # Organize data

    return hist, (w_ref_energy_data, wo_ref_energy_data)


if __name__ == "__main__":
    np.random.seed(428)

    gates = {
        "2-OR": [lambda x, y, z: (x | y) == z, 3, 0],
        "2-AND": [lambda x, y, z: (x & y) == z, 3, 0],
        "2-XOR": [lambda x, y, z: (x ^ y) == z, 3, 1],
        "1-ADDER": [lambda a0, b0, s0, c: ((a0 ^ b0) == s0 and (a0 & b0) == c), 4, 0],
        "3-OR": [lambda x, y, z, a: (x | y | z) == a, 4, 1],
    }

    Kl = 1
    Kc = 1
    Kt = 0.01
    Kt_ratio = 100
    t_span = (0, 10)
    dt0 = 0.01
    anneal = True

    for gate_name, [gate_func, n_io_var, n_aux_osc] in gates.items():
        model, j_mat = synthesize_general(
            gate_func,
            n_io_var=n_io_var,
            n_aux_osc=n_aux_osc,
            symmetric=True,
            constrain_coupling=0,
            constrain_energy_threshold=False,
        )

        # np.random.seed(428)
        hist, (w_ref_energy_data, wo_ref_energy_data) = validate_synthesis(
            model,
            j_mat,
            gate_func,
            n_io_var=n_io_var,
            n_aux_osc=n_aux_osc,
            Kc=Kc,
            Kl=Kl,
            Kt=Kt,
            Kt_ratio=Kt_ratio,
            plot=True,
            anneal=anneal,
            t_span=t_span,
            dt0=dt0,
        )
