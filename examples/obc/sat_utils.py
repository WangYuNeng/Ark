from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import spec_optimization as opt_spec

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.optimization.base_module import BaseAnalogCkt
from ark.specification.trainable import Trainable, TrainableMgr

(FALSE_PHASE, TRUE_PHASE, BLUE_PHASE) = (0, 2 / 3, 4 / 3)


def locking_3x(x, lock_strength: float):
    return lock_strength * jnp.sin(3 * jnp.pi * x)


@dataclass
class Clause:
    var0: int
    var1: int
    var2: int

    def __iter__(self):
        return iter((self.var0, self.var1, self.var2))

    def __getitem__(self, idx):
        return (self.var0, self.var1, self.var2)[idx]


@dataclass
class Problem:
    clauses: list[Clause]

    def __iter__(self):
        return iter(self.clauses)

    def __getitem__(self, idx):
        return self.clauses[idx]

    def __len__(self):
        return len(self.clauses)


@dataclass
class Assignment:
    vars: list[int]

    def __iter__(self):
        return iter(self.vars)

    def __getitem__(self, idx):
        return self.vars[idx]


@dataclass
class SATOscNetwork:
    """Class to store all the nodes and edges of the 3-SAT graph.

    Attributes:
        clause_oscs: list[list[CDGNode]], Clause oscillators. Length is the number of clauses and each entry is a list of 6 oscillators.
        var_oscs: list[tuple[CDGNode, CDGNode]],Variable oscillators. Length is the number of variables and each entry is a tuple of 2 oscillators,
            representing the negative and positive variable oscillators.
        base_oscs: tuple[CDGNode, CDGNode, CDGNode], Base oscillators. Tuple of 3 oscillators representing the False, True and Blue oscillators.
        clause_cpls: list[list[CDGEdge]], Clause coupling edges. Length is the number of clauses and each entry is a list of 5 coupling edges
            within each one of the clause oscillators.
        var_cpls: list[CDGEdge], Variable coupling edges. Length is the number of variables and each entry is the coupling edge between the negative
            and positive variable oscillators.
        base_to_clause_cpls: list[list[CDGEdge]], Coupling edges between the base oscillators and the clause oscillators. Length is the number of clauses
            and each entry is a list of 5 coupling edges. The first 4 edges are from True oscillator to the 1st, 2nd, 4th and 6th oscillators of the clause
            and the last edge is from False oscillator to the 5th oscillator of the clause.
        blue_to_var_cpls: list[tuple[CDGEdge, CDGEdge]], Coupling edges between the Blue oscillator and the variable oscillators. Length is the number of variables
            and each entry is a tuple of 2 coupling edges, from the Blue oscillator to the negative and positive variable oscillators, respectively.
        var_clause_cpls: list[list[tuple[tuple[CDGEdge, CDGEdge, CDGEdge], tuple[CDGEdge, CDGEdge, CDGEdge]]]], Coupling edges between the variable oscillators and
            the clause oscillators. Length is the number of variables and each entry is a list of length # of clauses. Each entry of the list is a 2-tuple of
            switchable coupling edges, from the negative and positive variable oscillators to the 1st, 4th and 6th oscillators of the clause, respectively.
        clause_lock_cpls: list[list[CDGEdge]], Coupling edges for locking the clause oscillators. Length is the number of clauses and each entry is a list of 6 coupling
            edges within each one of the clause oscillators.
        var_lock_cpls: list[tuple[CDGEdge, CDGEdge]], Coupling edges for locking the variable oscillators. Length is the number of variables and each entry is a tuple
            of 2 coupling edges, from the negative and positive variable oscillators to themselves.
        var_clause_cpls_args_idx: list[list[tuple[tuple[int, int, int], tuple[int, int, int]]]], The indexes of the switches between the variable oscillators and
            the clause oscillators in the arguments passing to the optimizer.
    """

    clause_oscs: list[list[CDGNode]]
    var_oscs: list[tuple[CDGNode, CDGNode]]
    base_oscs: tuple[CDGNode, CDGNode, CDGNode]

    clause_cpls: list[list[CDGEdge]]
    var_cpls: list[CDGEdge]
    base_to_clause_cpls: list[list[CDGEdge]]
    blue_to_var_cpls: list[tuple[CDGEdge, CDGEdge]]
    var_clause_cpls: list[
        list[tuple[tuple[CDGEdge, CDGEdge, CDGEdge], tuple[CDGEdge, CDGEdge, CDGEdge]]]
    ]
    clause_lock_cpls: list[list[CDGEdge]]
    var_lock_cpls: list[tuple[CDGEdge, CDGEdge]]

    var_clause_cpls_args_idx: list[
        list[tuple[tuple[int, int, int], tuple[int, int, int]]]
    ]

    def set_var_clause_cpls_args_idx(self, model: BaseAnalogCkt):
        self.var_clause_cpls_args_idx = [
            [
                tuple(
                    [
                        tuple(
                            [model.switch_to_args_id(sw.name) for sw in neg_switches]
                        ),
                        tuple(
                            [model.switch_to_args_id(sw.name) for sw in pos_switches]
                        ),
                    ]
                )
                for neg_switches, pos_switches in clauses
            ]
            for clauses in self.var_clause_cpls
        ]

    def problem_to_switch_array(self, problem: Problem):
        """
        Build the swtich array based on the input clauses.

        All switches are initialized to False, and only turned on for the the variable oscillators connecting
        to the clause oscillators.

        Args:
            clauses (list[tuple[int, int, int]]): List of clauses, each clause is a tuple of 3 integers
                representing the variable indices (1-indexed) in the clause.
        """
        assert (
            self.var_clause_cpls_args_idx is not None
        ), "mapping from switches to arguments indexes is not set"
        assert len(problem) == len(
            self.clause_oscs
        ), "Number of clauses must match the number of clause oscillators"
        len_switch_args = 6 * len(self.var_oscs) * len(self.clause_oscs)
        switch_arr = [0 for _ in range(len_switch_args)]
        for clause_id, clause in enumerate(problem):
            for nth_var, signed_var in enumerate(clause):
                # Get the index of the variable oscillator
                var_idx = abs(signed_var) - 1
                is_pos = signed_var > 0
                switch_idx = self.var_clause_cpls_args_idx[var_idx][clause_id][is_pos][
                    nth_var
                ]
                # Set the switch to True
                switch_arr[switch_idx] = 1
        return switch_arr


def create_3sat_graph(n_vars: int, n_clauses: int, trainable_mgr: TrainableMgr):
    """
    Create a configurable 3-SAT graph with the given # of variables and clauses.

    variable oscillators have switchable coupling to the 1st, 4th and 6th oscillators of each
    clause.

    Args:
        n_vars (int): Number of variables.
        n_clauses (int): Number of clauses.
    """

    sat_graph = CDG()
    # Good solution found for 3 variables and 7 clauses
    # weight_str = "0.83653987  1.20517709 -2.07106898  0.05542634  2.28063601  -0.1 -2.16215055 -2.08885618 -0.68421482"
    weight_str = "1 1 -1 1 1 -1 -1 -1 -1"
    ws = [float(w) for w in weight_str.strip("[]").split()]
    # Parameters for oscillators and couplings
    var_osc_args = {
        "lock_fn": locking_3x,
        "osc_fn": opt_spec.coupling_fn,
        "lock_strength": trainable_mgr.new_analog(init_val=ws[0]),
        "cpl_strength": trainable_mgr.new_analog(init_val=ws[1]),
    }
    var_cpl_args = {
        "k": trainable_mgr.new_analog(init_val=ws[2]),
    }
    clause_osc_args = {
        "lock_fn": locking_3x,
        "osc_fn": opt_spec.coupling_fn,
        "lock_strength": trainable_mgr.new_analog(init_val=ws[3]),
        "cpl_strength": trainable_mgr.new_analog(init_val=ws[4]),
    }
    clause_cpl_args = {
        "k": trainable_mgr.new_analog(init_val=ws[5]),
    }
    blue2var_cpl_args = {
        "k": trainable_mgr.new_analog(init_val=ws[6]),
    }
    base2clause_cpl_args = {
        "k": trainable_mgr.new_analog(init_val=ws[7]),
    }
    var2clause_cpl_args = {
        "k": trainable_mgr.new_analog(init_val=ws[8]),
    }

    # Create True, False, Blue oscillators
    f_osc, t_osc, b_osc = (
        opt_spec.FixedSource(phase=FALSE_PHASE),
        opt_spec.FixedSource(phase=TRUE_PHASE),
        opt_spec.FixedSource(phase=BLUE_PHASE),
    )
    base_oscs = (f_osc, t_osc, b_osc)

    # Create varaible oscillators var_oscs[i][0] for -(i+1) and var_oscs[i][1] for +(i+1)
    # var_cpls[i][0] for -(i+1) to Blue, var_cpls[i][1] for +(i+1) to Blue, and var_cpls[i][2]
    # for -(i+1) to +(i+1)
    var_oscs, var_cpls, var_lock_cpls = [], [], []
    blue_to_var_cpls = []
    for _ in range(n_vars):
        oscs = tuple([opt_spec.Osc_modified(**var_osc_args) for _ in range(2)])
        self_cpls = tuple([opt_spec.SelfCpl() for _ in range(2)])
        osc_cpl = opt_spec.Coupling(**var_cpl_args)
        blue_cpls = tuple([opt_spec.Coupling(**blue2var_cpl_args) for _ in range(2)])

        # Self connection for locking
        for osc, cpl in zip(oscs, self_cpls):
            sat_graph.connect(cpl, osc, osc)

        # Negative coupling between pos and neg variable oscillators
        # Two variable oscillators should be out of phase
        sat_graph.connect(osc_cpl, oscs[0], oscs[1])

        # Connect Blue oscillator to variable oscillators
        sat_graph.connect(blue_cpls[0], b_osc, oscs[0])
        sat_graph.connect(blue_cpls[1], b_osc, oscs[1])

        var_oscs.append(oscs)
        var_cpls.append(osc_cpl)
        var_lock_cpls.append(self_cpls)
        blue_to_var_cpls.append(blue_cpls)

    # Create clause oscillators and connect them to the variable oscillators
    clause_oscs, clause_cpls, clause_lock_cpls = [], [], []
    base_to_clause_cpls = []
    var_clause_cpls = [[] for _ in range(n_vars)]
    clause_var_conn_idx = [0, 3, 5]
    for _ in range(n_clauses):
        oscs = [opt_spec.Osc_modified(**clause_osc_args) for _ in range(6)]
        cpls = [opt_spec.Coupling(**clause_cpl_args) for _ in range(5)]
        b2c_cpls = [opt_spec.Coupling(**base2clause_cpl_args) for _ in range(5)]
        self_cpls = [opt_spec.SelfCpl() for _ in range(6)]

        # Connect internal clause oscillators
        sat_graph.connect(cpls[0], oscs[0], oscs[1])
        sat_graph.connect(cpls[1], oscs[1], oscs[2])
        sat_graph.connect(cpls[2], oscs[2], oscs[3])
        sat_graph.connect(cpls[3], oscs[2], oscs[4])
        sat_graph.connect(cpls[4], oscs[4], oscs[5])

        # Connect base oscillators to clause oscillators
        sat_graph.connect(b2c_cpls[0], t_osc, oscs[0])
        sat_graph.connect(b2c_cpls[1], t_osc, oscs[1])
        sat_graph.connect(b2c_cpls[2], t_osc, oscs[3])
        sat_graph.connect(b2c_cpls[3], t_osc, oscs[5])
        sat_graph.connect(b2c_cpls[4], f_osc, oscs[4])

        # Connect clause oscillators to themselves for locking
        for osc, cpl in zip(oscs, self_cpls):
            sat_graph.connect(cpl, osc, osc)

        clause_oscs.append(oscs)
        clause_cpls.append(cpls)
        base_to_clause_cpls.append(b2c_cpls)
        clause_lock_cpls.append(self_cpls)

        # Connect variable oscillators to clause oscillators
        # Both the positive and negative variable oscillators connect to the 1st, 4th and 6th
        # oscillators of the clause with switchable coupling
        for i in range(n_vars):
            v2c_edges = tuple(
                [
                    tuple(
                        [
                            opt_spec.Coupling(switchable=True, **var2clause_cpl_args)
                            for _ in range(3)
                        ]
                    )
                    for _ in range(2)
                ]
            )
            for j, idx in enumerate(clause_var_conn_idx):
                # Connect the negative variable oscillator to the clause oscillator
                sat_graph.connect(v2c_edges[0][j], var_oscs[i][0], oscs[idx])
                # Connect the positive variable oscillator to the clause oscillator
                sat_graph.connect(v2c_edges[1][j], var_oscs[i][1], oscs[idx])
            var_clause_cpls[i].append(v2c_edges)

    sat_network = SATOscNetwork(
        clause_oscs=clause_oscs,
        var_oscs=var_oscs,
        base_oscs=base_oscs,
        clause_cpls=clause_cpls,
        var_cpls=var_cpls,
        base_to_clause_cpls=base_to_clause_cpls,
        blue_to_var_cpls=blue_to_var_cpls,
        var_clause_cpls=var_clause_cpls,
        clause_lock_cpls=clause_lock_cpls,
        var_lock_cpls=var_lock_cpls,
        var_clause_cpls_args_idx=None,
    )

    return sat_graph, sat_network
