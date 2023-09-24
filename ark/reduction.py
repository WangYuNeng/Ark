"""
Reduction functions for the dynamical system.
"""
import ast
from dataclasses import dataclass


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
    name: str


SUM = Reduction(name="sum", ast_op=ast.Add(), ast_switch=ast.Mult())
PRODUCT = Reduction(name="mul", ast_op=ast.Mult(), ast_switch=ast.Pow())
