# Demonstrate syntheizing function with oscillators

from itertools import product
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from cvc5.pythonic import Const, Int, Ints, Or, Reals, Solver, sat
from scipy.integrate import solve_ivp


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
    return 1 if p == 1 else 0


def random_simulation(
    J: np.ndarray,
    n_sim=256,
    Kc=1,
    Kl=1,
    super_harmonic=2,
    t_span=(0, 2),
    dt=0.01,
    anneal=False,
):
    n_osc = J.shape[0]
    fix_indx = np.array([3])  # Index of the fixed oscillator

    def obc_ode(t, y):
        dydt = np.zeros(n_osc)
        y_diff = y[:, None] - y[None, :]
        coupling = Kc * np.sum(J * np.sin(y_diff), axis=1)
        # Exponential schedule
        if anneal:
            Kl_anneal = Kl * (1 - np.exp(-0.1 * t))
        else:
            Kl_anneal = Kl
        locking = Kl_anneal * np.sin(super_harmonic * y)
        dydt = coupling - locking
        dydt[fix_indx] = 0  # Fix one oscillator
        return dydt

    results = []
    for _ in range(n_sim):
        y0 = np.random.uniform(-np.pi, np.pi, n_osc)
        y0[fix_indx] = np.pi  # Fix the reference oscillator
        phases = solve_ivp(
            obc_ode,
            t_span,
            y0,
            t_eval=[t_span[1]],
            method="RK45",
        ).y

        # Rectify to 0, 2pi/super_harmonic, 4pi/super_harmonic, ... 2pi
        rect_vals = [
            (2 * np.pi / super_harmonic) * i for i in range(super_harmonic + 1)
        ]
        phases = np.array(
            [
                rect_vals[np.argmin(np.abs(rect_vals - (v % (2 * np.pi))))]
                for v in phases[:, -1]
            ]
        )
        # Map 2pi back to 0
        phases = np.where(phases == 2 * np.pi, 0, phases)
        # print("Final phases:", phases)

        results.append(phases)
    return np.array(results)


def synthesize_general(
    logic_fn: Callable, n_free_osc: int, symmetric: bool = True, n_coupling: int = 0
):

    def energy_fn(J_max: np.ndarray, v: np.ndarray):
        n_var = len(v)
        assert J_max.shape == (n_var, n_var)

        # Pairwise products
        v_diff = v[:, None] * v[None, :]

        # Zero diagonal
        v_diff = v_diff * (1 - np.eye(n_var))
        return (J_max * v_diff).sum()

    # 5 oscillators are input/output, 1 , rest are free variables
    n_osc = n_free_osc + 4
    phases = [-1, 1]
    fixed_oscs = [1]  # Two fixed oscillators as references

    constraints = []

    # NxN coupling matrix
    # J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J_max = np.array(J).reshape((n_osc, n_osc))

    # Auxiliary variable: minimum energy
    e_min = Int("e_min")

    # Enumerate all free boolean values
    free_val_tab = [list(v) for v in product(*[phases for _ in range(n_free_osc)])]

    for xb, yb, ob in [
        (_x, _y, _o) for _x in range(2) for _y in range(2) for _o in range(2)
    ]:
        xyo = bool_to_phase([xb, yb, ob])
        possible_min_energy_states = []
        for free_vals in free_val_tab:
            v = np.array(xyo + fixed_oscs + free_vals)
            e = energy_fn(J_max, v)
            if ob == logic_fn(xb, yb):
                # Solution states have energy at least e_min
                constraints.append(e >= e_min)
                possible_min_energy_states.append(e)
            else:
                # Non-solution states have energy strictly greater than e_min
                constraints.append(e > e_min)
        if possible_min_energy_states:
            # At least one solution state has energy equal to e_min
            constraints.append(Or(*[e == e_min for e in possible_min_energy_states]))

    # Symmetry and zero diagonal constraints
    for i in range(n_osc):
        constraints.append(J[i * n_osc + i] == 0)
        if symmetric:
            for j in range(i + 1, n_osc):
                constraints.append(J[i * n_osc + j] == J[j * n_osc + i])

    # Constraint the number of non-zero couplings
    if n_coupling > 0:
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
            sum(coupling_indicators) <= n_coupling
        )  # Limit the number of couplings

    # Solve
    s = Solver()
    s.add(*constraints)
    if s.check() != sat:
        print("UNSAT, no solution found")

    else:
        m = s.model()
        j_mat = []
        print("Minimum energy:", m[e_min])

        print("Coupling matrix:")

        for i in range(n_osc):
            for j in range(n_osc):
                j_mat.append(m[J[i * n_osc + j]])
        print(np.array(j_mat).reshape((n_osc, n_osc)))

        # List the truth table and corresponding energies
        print("Truth table and energies:")
        for xb, yb, ob in [
            (_x, _y, _o) for _x in range(2) for _y in range(2) for _o in range(2)
        ]:
            xyo = bool_to_phase([xb, yb, ob])
            energies = []
            for free_vals in free_val_tab:
                v = np.array(xyo + fixed_oscs + free_vals)
                e = energy_fn(J_max, v)
                energies.append(m[e])
            print(f"x={xb}, y={yb}, o={ob} => energies: {energies}")

        # Validate with simulation
        J_val = (
            np.array([val.as_long() for val in j_mat])
            .reshape((n_osc, n_osc))
            .astype(float)
        )
        sim_results = random_simulation(J_val, n_sim=2**10, Kl=1, Kc=1)

        # Plot the sim result histogram -- count how many times each input/output combination occurs
        hist = {(x, y, o): 0 for x in [0, 1] for y in [0, 1] for o in [0, 1]}
        for res in sim_results:
            x, y, o = [int(bool(p)) for p in res[:3]]
            key = (x, y, o)
            hist[key] += 1
        print("Simulation results (x, y, o): count")
        for key, count in hist.items():
            print(f"{key}: {count}")
        plt.bar(
            range(len(hist)), hist.values(), tick_label=[str(k) for k in hist.keys()]
        )
        plt.xlabel("(x, y, o)")
        plt.ylabel("Count")
        plt.title("Simulation Results Histogram")
        plt.show()


if __name__ == "__main__":

    # synthesize_logic(lambda x, y: x | y)  # OR
    # synthesize_logic(lambda x, y: x & y)  # AND
    # synthesize_logic(lambda x, y: x ^ y)  # XOR
    # synthesize_logic(lambda x, y: (x & (not y)) | ((not x) & y))  # XNOR
    # synthesize_logic(lambda x, y: not (x & y))  # NAND
    # synthesize_logic(lambda x, y: not (x & y))  # NAND

    synthesize_general(lambda x, y: x | y, 0)
    synthesize_general(lambda x, y: not (x | y), 0)
    synthesize_general(lambda x, y: x & y, 0)
    synthesize_general(lambda x, y: not (x & y), 0)
    synthesize_general(lambda x, y: not x, 0)

    synthesize_general(lambda x, y: x ^ y, 0)  # XOR, no solution w/o free osc
    synthesize_general(lambda x, y: x ^ y, 1)  # XOR
