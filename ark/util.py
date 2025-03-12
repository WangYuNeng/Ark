import ast
from copy import copy
from functools import partial
from typing import Any, Callable, Iterable

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


def mk_var_generator(name: str):
    return lambda: ast.Name(id=name)


def mk_assign(target: ast.expr, value: ast.expr):
    """target = value"""
    return ast.Assign(
        targets=[set_ctx(target, ast.Store())],
        value=set_ctx(value, ast.Load()),
    )


def mk_call(fn: ast.expr, args: list[ast.expr]):
    """fn(*args)"""
    return ast.Call(
        func=fn,
        args=args,
        keywords=[],
    )


def mk_tuple(elts: list[ast.Name | ast.Constant]):
    return ast.Tuple(elts=elts)


def mk_list(lst: list[ast.Name | ast.Constant]):
    return ast.List(elts=lst)


def mk_arr_access(lst: ast.Name, idx: ast.expr):
    return ast.Subscript(
        value=lst,
        slice=idx,
    )


def mk_list_val_expr(lst: Iterable):
    """make the list value to be expressions"""
    lst_expr = []
    for val in lst:
        if isinstance(val, ast.expr):
            lst_expr.append(val)
        elif isinstance(val, (int, float)) or val is None:
            lst_expr.append(ast.Constant(value=val))
        else:
            raise ValueError(f"Unknown type {type(val)} to be converted in the list")
    return lst_expr


def mk_jnp_call(args: list, call_fn: str):
    """jnp.call_fn(*args)"""

    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="jnp"), attr=call_fn),
        args=mk_list_val_expr(args),
        keywords=[],
    )


def mk_jax_random_call(args: list, call_fn: str):
    """jax.random.call_fn(*args)"""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="jax"), attr="random"), attr=call_fn
        ),
        args=mk_list_val_expr(args),
        keywords=[],
    )


def mk_jnp_arr_access(arr: ast.Name, idx: ast.expr):
    """arr.at[idx]"""
    return ast.Subscript(
        value=ast.Attribute(value=arr, attr="at"),
        slice=idx,
    )


def mk_jnp_assign(arr: ast.Name, idx: ast.expr, val: ast.Name | ast.Constant):
    """
    arr = arr.at[idx].set(val)
    """
    return mk_assign(
        target=arr,
        value=ast.Call(
            func=ast.Attribute(
                value=mk_jnp_arr_access(copy(arr), idx),
                attr="set",
            ),
            args=[val],
            keywords=[],
        ),
    )


def mk_jnp_scatter_gather(arr_size: int, idx: ast.expr, val: ast.Expr, gather: str):
    """
    arr.at[idx].op(val)
    """
    scatter_base_zero = mk_jnp_call(
        args=[ast.Constant(value=arr_size)],
        call_fn="zeros" if gather == "add" else "ones",
    )
    scatter_at = ast.Attribute(value=scatter_base_zero, attr="at")
    return ast.Call(
        func=ast.Attribute(
            value=mk_arr_access(lst=scatter_at, idx=idx),
            attr=gather,
        ),
        args=[val],
        keywords=[],
    )
