from typing import Generator, Optional

import jax.numpy as jnp
import numpy as np
from sat_utils import Clause, Problem, SATOscNetwork, create_3sat_graph


def sat_3var7clauses_data() -> tuple[list[Problem], list[Clause]]:
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
        solutions.append(Clause(*[-var for var in all_clauses[removed_clause_idx]]))
        all_clauses.pop(removed_clause_idx)
        sat_probs.append(Problem(all_clauses))
    return sat_probs, solutions


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

    def __iter__(
        self,
    ) -> Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:
        """Generate 1. random intial states for the OBC network, 2. the switch array for the SAT problem,
        3. the solution for the SAT problem if provided.

        The initial state is a random array of shape (batch_size, num_oscillators) with values in {0, 2}
        (0~2pi normalized by pi).
        The switch array corresponds to the clauses in the SAT problem.
        The solution is an array of shape (batch_size, num_vars) which is a satisfying assignment.

        Args:
            batch_size (int): The batch size for the dataloader.

        Returns:
            Generator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]: initial states, switch array, and solution.
        """

        osc_network = self.osc_network
        batch_size = self.batch_size
        n_oscillators = 2 * len(osc_network.var_oscs) + 6 * len(osc_network.clause_oscs)

        while True:
            initial_states = np.random.rand(batch_size, n_oscillators) * 2
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
            yield (
                jnp.array(initial_states),
                jnp.array(switch_arrs),
                jnp.array(solutions) if self.sat_solutions else None,
            )
