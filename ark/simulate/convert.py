"""
Functions and classes for converting between Ark specification and graphs and simulation ready data.
"""
from typing import Tuple, Callable, List, Protocol

from ark.cdg.cdg import CDG
from ark.compiler import ArkCompiler
from ark.rewrite import SympyRewriteGen
from ark.specification.specification import CDGSpec

import sympy as sp
import equinox as eqx
import jax.numpy as jnp
from jax import Array


def _tuple_to_sympy_eq(pair: Tuple[sp.Symbol, sp.Expr]) -> sp.Eq:
    """Turns tuple of derivative + sympy expression into a single sympy equation."""
    if (var_name := pair[0].name).startswith("ddt_"):
        symbol = sp.symbols(var_name[4:])
        equation = sp.Eq(sp.Derivative(symbol, sp.symbols("time")), pair[1])
        return equation
    else:
        raise ValueError("Not a derivative expression.")


def _cleanup_diffeq(pair: Tuple[sp.Expr, sp.Expr]) -> sp.Eq:
    eq = _tuple_to_sympy_eq(pair)
    return eq


def get_variables(equation: sp.Eq) -> Tuple[sp.Symbol, List[sp.Symbol]]:
    # Check that there is only one free symbol on the left hand side
    assert (
        len(equation.lhs.free_symbols) == 1
    ), f"Equation {equation} has more than one free symbol on the left hand side."

    return list(equation.lhs.free_symbols)[0], list(equation.rhs.free_symbols)


class DynamicCallable(Protocol):
    def __call__(self, array: Array, parameters: dict) -> Array:
        ...


class JaxFunction:
    array_variables: List[sp.Symbol]
    "Sent by array in this order"
    parameter_variables: List[sp.Symbol]
    "Sent by dictionary"
    jax_function: DynamicCallable

    def __init__(
        self,
        array_variables: List[sp.Symbol],
        parameter_variables: List[sp.Symbol],
        jax_function: Callable[[Array, dict], Array],
    ):
        self.array_variables = array_variables
        self.parameter_variables = parameter_variables
        self.jax_function = jax_function

    def __call__(self, array: Array, parameters: dict) -> Array:
        return self.jax_function(array=array, parameters=parameters)


def create_jax_function(equations: List[sp.Eq]) -> JaxFunction:
    eq_vars = list(map(get_variables, equations))
    lhs_vars = [var for var, _ in eq_vars]
    all_jax_functions = [sp.lambdify(args=eq.rhs.free_symbols, expr=eq.rhs, modules='jax') for eq in equations]
    all_rhs_vars = set([s for eq in equations for s in list(eq.rhs.free_symbols)])


    def jax_function(vector: Array, parameters: dict) -> Array:
        all_parameters = {}
        for label, val in zip(lhs_vars, vector):
            all_parameters[label.name] = val
        all_parameters.update(parameters)
        result = jnp.zeros(len(vector))
        for i, jax_fn in enumerate(all_jax_functions):
            result = result.at[i].set(jax_fn(**all_parameters))
        return result
    return JaxFunction(array_variables=lhs_vars,
                       parameter_variables=list(all_rhs_vars),
                       jax_function=jax_function)


class DynamicalSystem:
    """A universal dynamical system description."""

    @staticmethod
    def from_ark(cdg: CDG, spec: CDGSpec):
        # Compile the equation to a set of tuples of sympy expressions
        compiler = ArkCompiler(rewrite=SympyRewriteGen())
        sympy_eqs = compiler.compile_sympy(cdg_spec=spec, cdg=cdg, help_fn=[])

        # Turn these into real differential equations with variables
        sympy_eqs = [_cleanup_diffeq(eq) for eq in sympy_eqs]

        # Create jax function with retrieval

        # Organize nodes for retrieving parameters
        node_dict = {node.name: node for node in cdg.nodes}

    def __init__(self):
        pass

    def equinox_module(self):
        pass


class EquinoxSystem(eqx.Module):
    def __init__(self):
        pass
