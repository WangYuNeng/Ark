from abc import ABC, abstractmethod
from pysmt.shortcuts import Symbol, And, GE, LE, Plus, Equals, Int, get_model
from pysmt.typing import INT
import numpy as np
from ark.specification.range import Range


class Solver(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def solve_validation_matrix(self, matrix, constraints):
        pass

class SMTSolver(Solver):

    def solve_validation_matrix(self, matrix: list[list[bool]], constraints: list[Range]):
        """
        Solve the validation matrix using SMT solver.
        TODO: Prettify the code, e.g., rename "constraints"
        """
        variables = np.array([[Int(0) for _ in row] for row in matrix])
        row_constraints = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val == 1:
                    variables[i, j] = Symbol('var_{}_{}'.format(i, j), INT)
            row_constraints.append(Equals(Plus(variables[i, :].tolist()), Int(1)))
        col_constraints = []
        for j, constraint in enumerate(constraints):
            summation = Plus(variables[:, j].tolist())
            col_constraints.append(constraint.to_smt(summation))
        problem = And(*row_constraints, *col_constraints)
        model = get_model(problem)
        if model:
            return True
        else:
            return False
