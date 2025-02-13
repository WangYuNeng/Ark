import ast
from functools import partial
from typing import Any, Callable

import sympy


def inspect_func_name(func: Callable[..., Any]) -> str:
    """Return the name of the function.

    Args:
        func (Callable[..., Any]): The function to inspect.

    Returns:
        str: The name of the function.
    """
    if isinstance(func, partial):
        return func.func.__name__
    return func.__name__


def set_ctx(expr: ast.expr, ctx: ast.expr_context) -> ast.expr:
    """set the context of the given node and its children to the given context"""
    for nch in ast.iter_child_nodes(expr):
        if isinstance(nch, ast.expr):
            setattr(nch, "ctx", ctx)
            set_ctx(nch, ctx)

    if isinstance(expr, ast.expr):
        setattr(expr, "ctx", ctx)
    return expr


def concat_expr(
    exprs: list[ast.expr] | list[sympy.Expr], operator: ast.operator | sympy.Expr
) -> ast.operator | sympy.Expr:
    """concatenate expressions with the given operator"""
    if isinstance(operator, ast.operator):
        if len(exprs) == 1:
            return exprs[0]
        if len(exprs) == 2:
            return ast.BinOp(left=exprs[0], op=operator, right=exprs[1])
        return ast.BinOp(
            left=exprs[0], op=operator, right=concat_expr(exprs[1:], operator)
        )
    else:
        return operator(*exprs)
