import os
from typing import Generator, Optional

import jax.numpy as jnp
import numpy as np
from sat_utils import Assignment, Clause, Problem, SATOscNetwork, parse_cnf_file


def sat_kvar_exact_assignment_clauses_with_redundant_data(
    k: int, d: int
) -> tuple[list[Problem], list[Assignment]]:
    """
    Generate a 3-SAT problem with k+d clauses and k variables.
    Clause i is (xi or xi or xi).

    d clauses are redundant clauses that do not affect the satisfiability of the problem.

    Args:
        k (int): # of variables and clauses in the 3-SAT problem.

    Returns:
        tuple[list[Problem], list[Clause]]: A tuple containing a list of SAT problems and a list of clauses
        representing the exact satisfying assignment for each problem.
    """

    clauses = [Clause(-i, -i, -i) for i in range(1, k + 1)]
    # Add redundant clauses
    clauses += [
        Clause(
            *(
                np.random.choice(
                    [-i for i in range(1, k + 1)], size=3, replace=True
                ).tolist()
            )
        )
        for _ in range(d)
    ]
    solutions = [Assignment([-i for i in range(1, k + 1)])]
    sat_prob = Problem(clauses)
    return [sat_prob], solutions


def sat_2var3clauses_data() -> tuple[list[Problem], list[Assignment]]:
    """
    Generate a 3-SAT problem with 3 clauses and 2 variables.
    Enumerate all possible combinations of exactly 1 satisfying assignment.
    """
    sat_probs, solutions = [], []
    for removed_clause_idx in range(3):
        all_clauses = [Clause(i, j, j) for i in [-1, 1] for j in [-2, 2]]
        # Remove the clause at the specified index
        solutions.append(
            Assignment([-var for var in all_clauses[removed_clause_idx][:-1]])
        )
        all_clauses.pop(removed_clause_idx)
        sat_probs.append(Problem(all_clauses))
    return sat_probs, solutions


def sat_3var7clauses_data() -> tuple[list[Problem], list[Assignment]]:
    """
    Generate a 3-SAT problem with 7 clauses and 3 variables.
    Enumerate all possible combinations of exactly 1 satisfying assignment.
    """
    sat_probs, solutions = [], []
    for removed_clause_idx in range(8):
        all_clauses = [
            Clause(i, j, k) for i in [-1, 1] for j in [-2, 2] for k in [-3, 3]
        ]
        # Remove the clause at the specified index
        solutions.append(Assignment([-var for var in all_clauses[removed_clause_idx]]))
        all_clauses.pop(removed_clause_idx)
        sat_probs.append(Problem(all_clauses))
    return sat_probs, solutions


def sat_from_cnf_dir(dir_path: str) -> list[Problem]:
    """
    Load SAT problems from a directory containing CNF files.

    Args:
        dir_path (str): The path to the directory containing CNF files.

    Returns:
        list[Problem]: A list of SAT problems, each represented as a Problem object.
    """
    sat_probs = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".cnf"):
            file_path = os.path.join(dir_path, file_name)
            sat_probs.append(parse_cnf_file(file_path))
    return sat_probs


def sat_random_clauses(n_vars: int, n_clauses: int, n_prob: int) -> list[Problem]:
    """
    Generate a list of random SAT problems with a specified number of variables and clauses.

    Args:
        n_vars (int): Number of variables in each SAT problem.
        n_clauses (int): Number of clauses in each SAT problem.
        n_prob (int): Number of SAT problems to generate.

    Returns:
        list[Problem]: A list of randomly generated SAT problems.
    """
    sat_probs = []
    for _ in range(n_prob):
        clauses = [
            Clause(
                *np.random.choice(
                    [i for i in range(-n_vars, 0)] + [i for i in range(1, n_vars + 1)],
                    size=3,
                    replace=True,
                ).tolist()
            )
            for _ in range(n_clauses)
        ]
        sat_probs.append(Problem(clauses))
    return sat_probs


class SATDataloader:
    """
    A dataloader to prepare the SAT problem for the OBC.

    Args:
        sat_probs: A list of SAT problems, each represented as a list of clauses.
        batch_size: The batch size for the dataloader.
        osc_network: The OBC network to be used for the SAT problem, must have the same # of variables and
            clauses as the SAT problem.
    """

    def __init__(
        self,
        batch_size: int,
        sat_probs: list[Problem],
        osc_network: SATOscNetwork,
        sat_solutions: Optional[list[Clause]] = None,
    ):
        assert all(
            len(sat_probs[0]) == len(sat_probs[i]) for i in range(1, len(sat_probs))
        ), "All SAT problems must have the same number of clauses."
        assert len(osc_network.clause_oscs) == len(
            sat_probs[0]
        ), "The OBC network must have the same number of clauses as the SAT problem."
        assert len(osc_network.var_oscs) >= max(
            abs(var) for sat_prob in sat_probs for clause in sat_prob for var in clause
        ), "The OBC network must have the same number of variables as the SAT problem."

        self.batch_size = batch_size
        self.sat_probs = sat_probs
        self.osc_network = osc_network
        self.sat_solutions = sat_solutions

        self.probs_switch_arrs = [
            osc_network.problem_to_switch_array(prob) for prob in sat_probs
        ]

        self.probs_adjacency_matrices = [
            osc_network.problem_to_adjacency_matrix(prob) for prob in sat_probs
        ]

    def __iter__(
        self,
    ) -> Generator[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray],
        None,
        None,
    ]:
        """Generate 1. random intial states for the OBC network, 2. the switch array for the SAT problem,
        3. the solution for the SAT problem if provided.

        The initial state is a random array of shape (batch_size, num_oscillators) with values in {0, 2}
        (0~2pi normalized by pi).
        The switch array corresponds to the clauses in the SAT problem.
        The solution is an array of shape (batch_size, num_vars) which is a satisfying assignment.

        Args:
            batch_size (int): The batch size for the dataloader.

        Returns:
            Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]: initial states, switch array, solution (if given), and
            adjacency matrices.
        """

        osc_network = self.osc_network
        batch_size = self.batch_size
        n_oscillators = 2 * len(osc_network.var_oscs) + 6 * len(osc_network.clause_oscs)
        n_vars = len(osc_network.var_oscs)

        while True:
            initial_states = np.random.rand(batch_size, n_oscillators) * 2
            # initial states are equally spaced in [0, 2]
            # initial_states = np.linspace(0, 2, n_oscillators, endpoint=False)
            # initial_states = np.tile(initial_states, (batch_size, 1))
            sampled_prob_idx = np.random.choice(
                len(self.sat_probs), batch_size, replace=True
            )
            switch_arrs = np.array(
                [self.probs_switch_arrs[prob_idx] for prob_idx in sampled_prob_idx]
            )
            if self.sat_solutions:
                solutions = np.array(
                    [
                        [[var <= 0, var > 0] for var in self.sat_solutions[prob_idx]]
                        for prob_idx in sampled_prob_idx
                    ]
                )
                solutions = solutions.reshape(batch_size, 2 * len(osc_network.var_oscs))
            adj_matrices = np.array(
                [
                    self.probs_adjacency_matrices[prob_idx]
                    for prob_idx in sampled_prob_idx
                ]
            )
            probs = [
                self.sat_probs[prob_idx].to_list() for prob_idx in sampled_prob_idx
            ]
            yield (
                jnp.array(initial_states),
                jnp.array(switch_arrs),
                jnp.array(solutions) if self.sat_solutions else None,
                jnp.array(adj_matrices),
                n_vars,
                jnp.array(probs),
            )
