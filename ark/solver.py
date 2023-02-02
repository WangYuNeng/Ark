from abc import ABC, abstractmethod
from pysmt.shortcuts import Symbol, And, GE, LE, Plus, Equals, Int, get_model
from pysmt.typing import INT
import numpy as np

from ark.specification.constraint import Constraint, DegreeConstraint

class Solver(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def solve_validation_matrix(self, matrix, constraints):
        pass

class SMTSolver(Solver):

    def __init__(self) -> None:
        super().__init__()

    def solve_validation_matrix(self, matrix, constraints):

        variables = np.array([[Int(0) for _ in row] for row in matrix])
        row_constraints = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val == 1:
                    variables[i, j] = Symbol('var_{}_{}'.format(i, j), INT)
            row_constraints.append(Equals(Plus(variables[i, :].tolist()), Int(1)))
        col_constraints = []
        for j, constraint in enumerate(constraints):
            op, val = self.interpret_constraint(constraint)
            if op != None:
                col_constraints.append(op(Plus(variables[:, j].tolist()), val))
        problem = And(*row_constraints, *col_constraints)
        model = get_model(problem)
        if model:
            print(model)
            return True
        else:
            return False
    
    def interpret_constraint(self, constraint: Constraint):

        if isinstance(constraint, DegreeConstraint):
            expr = constraint.expr
            if expr == '*':
                return None, None
            elif expr.startswith('='):
                val = int(expr[1:])
                return Equals, Int(val)
            elif expr.startswith('<='):
                val = int(expr[2:])
                return LE, Int(val)
            else:
                raise NotImplementedError
        raise NotImplementedError