import jax.numpy as jnp
import numpy as np


def sat_3var7clauses_data():
    """
    Generate a 3-SAT problem with 7 clauses and 3 variables.
    Enumerate all possible combinations of exactly 1 satisfying assignment.
    """
    sat_probs = []
    for removed_clause_idx in range(8):
        all_clauses = [[i, j, k] for i in [-1, 1] for j in [-2, 2] for k in [-3, 3]]
        # Remove the clause at the specified index
        all_clauses.pop(removed_clause_idx)
        sat_probs.append(all_clauses)
    return sat_probs


class SATDataloader:

    def __init__(
        self,
        sat_probs: list[tuple[int, int, int]],
        batch_size: int,
    ):
        self.sat_probs = sat_probs
        self.batch_size = batch_size
        self.num_clauses = len(sat_probs[0])
        self.num_vars = len(sat_probs[0][0])
        self.num_samples = len(sat_probs)
        self.current_index = 0
