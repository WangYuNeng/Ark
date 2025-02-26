"""
Reduction functions for the dynamical system.
"""

import ast
from dataclasses import dataclass

import sympy


@dataclass
class Reduction:
    """
    Reduction base class.

    ast_op -- the ast operator corresponding to the reduction
    ast_switch -- the ast operator corresponding to the switch
        value switch_op switch_val should give
        - the identity element when switch is off
        - the value when switch is on
    """

    ast_op: ast.operator
    ast_switch: ast.operator
    sympy_op: type  # sympy.Mul is a type obj not a sympy.Expr obj (but sympy.Mul() is)
    sympy_switch: type
    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


SUM = Reduction(
    name="sum",
    ast_op=ast.Add(),
    ast_switch=ast.Mult(),
    sympy_op=sympy.Add,
    sympy_switch=sympy.Mul,
)
PRODUCT = Reduction(
    name="mul",
    ast_op=ast.Mult(),
    ast_switch=ast.Pow(),
    sympy_op=sympy.Mul,
    sympy_switch=sympy.Pow,
)
