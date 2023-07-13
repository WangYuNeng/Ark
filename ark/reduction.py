"""
Reduction functions for the dynamical system.
"""
import ast

class Reduction:
    """
    Reduction base class.
    """

    def ast_op(self) -> ast.operator:
        """return the ast operato correspond to the reduction"""
        raise NotImplementedError

class Sum(Reduction):
    """
    Summation
    """

    def ast_op(self) -> ast.operator:
        return ast.Add

class Product(Reduction):
    """
    Product
    """

    def ast_op(self) -> ast.operator:
        return ast.Mult


SUM = Sum()
PROD = Product()
