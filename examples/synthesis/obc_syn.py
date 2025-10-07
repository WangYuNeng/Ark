# Demonstrate syntheizing function with oscillators

from typing import Callable

import numpy as np
from cvc5.pythonic import Ints, Or, Reals, Solver, sat


def bool_to_phase(b):
    return 1 if b else -1


def phase_to_bool(p):
    return 1 if p == 1 else 0


def synthesize_logic(logic_fn: Callable):

    # Coupling
    c_xy, c_xr, c_yr, c_xo, c_yo, c_ro = Ints("c_xy c_xr c_yr c_xo c_yo c_ro")

    # Energy function
    def energy_fn(_x, _y, _o):
        return (
            c_xy * _x * _y
            + c_xr * _x
            + c_yr * _y
            + c_xo * _x * _o
            + c_yo * _y * _o
            + c_ro * _o
        )

    constraints = []
    # Phases are {-1, 1}
    phases = [-1, 1]
    energy_allstates = {
        (x, y, o): energy_fn(x, y, o) for x in phases for y in phases for o in phases
    }

    # Constraints for "OR" -- energy is minimzed when the logic values are in
    # the truth table of "OR"
    for x in range(2):
        for y in range(2):
            o = logic_fn(x, y)
            o_phase, x_phase, y_phase = (
                bool_to_phase(o),
                bool_to_phase(x),
                bool_to_phase(y),
            )
            e = energy_fn(x_phase, y_phase, o_phase)
            for phases, energy in energy_allstates.items():
                _x, _y, _o = phases
                _x_bool, _y_bool, _o_bool = (
                    phase_to_bool(_x),
                    phase_to_bool(_y),
                    phase_to_bool(_o),
                )
                if _o_bool != logic_fn(_x_bool, _y_bool):
                    constraints.append(e < energy)
                else:
                    constraints.append(e == energy)

    # Solve
    s = Solver()
    s.add(*constraints)
    if s.check() != sat:
        print("UNSAT, no solution found")

    else:
        m = s.model()
        print("c_xy =", m[c_xy])
        print("c_xr =", m[c_xr])
        print("c_yr =", m[c_yr])
        print("c_xo =", m[c_xo])
        print("c_yo =", m[c_yo])
        print("c_ro =", m[c_ro])
        print("c_ro =", m[c_ro])


def synthesize_general(logic_fn: Callable, n_free_osc: int, symmetric: bool = True):

    def energy_fn(J_max: np.ndarray, v: np.ndarray):
        n_var = len(v)
        assert J_max.shape == (n_var, n_var)

        # Pairwise products
        v_diff = v[:, None] * v[None, :]

        # Zero diagonal
        v_diff = v_diff * (1 - np.eye(n_var))
        return (J_max * v_diff).sum()

    # 5 oscillators are input/output, -1, and 1 , rest are free variables
    n_osc = n_free_osc + 5
    oscs = [1, -1]
    if n_free_osc > 0:
        oscs.extend(list(Ints(" ".join([f"v{i}" for i in range(n_free_osc)]))))

    constraints = []
    # Oscillator phases are {-1, 1}
    for osc in oscs:
        constraints.append(Or(osc == 1, osc == -1))

    # NxN coupling matrix
    # J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J = Ints(" ".join([f"J{i}{j}" for i in range(n_osc) for j in range(n_osc)]))
    J_max = np.array(J).reshape((n_osc, n_osc))

    energy_allstates = {}
    phases = [-1, 1]
    for x, y, o in [(_x, _y, _o) for _x in phases for _y in phases for _o in phases]:
        xb, yb, ob = phase_to_bool(x), phase_to_bool(y), phase_to_bool(o)
        v = np.array([x, y, o] + list(oscs))
        energy_allstates[(xb, yb, ob)] = energy_fn(J_max, v)

    for xb, yb in [(_x, _y) for _x in range(2) for _y in range(2)]:
        ob = logic_fn(xb, yb)
        o, x, y = (
            bool_to_phase(ob),
            bool_to_phase(xb),
            bool_to_phase(yb),
        )
        v = np.array([x, y, o] + list(oscs))
        e = energy_fn(J_max, v)
        for booleans, energy in energy_allstates.items():
            _xb, _yb, _ob = booleans
            if _ob != logic_fn(_xb, _yb):
                constraints.append(e < energy)
            else:
                constraints.append(e == energy)

    # Symmetry and zero diagonal constraints
    for i in range(n_osc):
        constraints.append(J[i * n_osc + i] == 0)
        if symmetric:
            for j in range(i + 1, n_osc):
                constraints.append(J[i * n_osc + j] == J[j * n_osc + i])

    # Solve
    s = Solver()
    s.add(*constraints)
    if s.check() != sat:
        print("UNSAT, no solution found")

    else:
        m = s.model()
        print("Free oscillator values:")
        v, j_mat = [], []
        for i in range(n_free_osc):
            v.append(m[oscs[i]])
        print(v)

        print("Coupling matrix:")

        for i in range(n_osc):
            for j in range(n_osc):
                j_mat.append(m[J[i * n_osc + j]])
        print(np.array(j_mat).reshape((n_osc, n_osc)))


if __name__ == "__main__":

    # synthesize_logic(lambda x, y: x | y)  # OR
    # synthesize_logic(lambda x, y: x & y)  # AND
    # synthesize_logic(lambda x, y: x ^ y)  # XOR
    # synthesize_logic(lambda x, y: (x & (not y)) | ((not x) & y))  # XNOR
    # synthesize_logic(lambda x, y: not (x & y))  # NAND
    # synthesize_logic(lambda x, y: not (x & y))  # NAND

    synthesize_general(lambda x, y: x | y, 0, False)  # OR with 1 free oscillators

    synthesize_general(
        lambda x, y: not (x & y), 0, False
    )  # NAND with 1 free oscillators

    synthesize_general(lambda x, y: not x, 0)  # Not with 0 free oscillators

    # for n_osc in range(20, 60):
    #     print(f"n_osc = {n_osc}")
    #     synthesize_general(lambda x, y: not x, n_osc, False)  # XOR
