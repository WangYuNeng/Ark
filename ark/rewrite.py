"""
Ast rewrite classes to generate expression from production rules
"""

import ast
from abc import ABC, abstractmethod
from typing import Callable

import sympy

from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, kw_name
from ark.util import set_ctx


class BaseRewriteGen(ABC):

    def __init__(self) -> None:
        self._attr_rn_fn = None

    @abstractmethod
    def visit(self, expr):
        raise NotImplementedError

    def set_attr_rn_fn(self, val: Callable[[str, str], str]) -> None:
        """set attribute renaming function"""
        self._attr_rn_fn = val


class RewriteGen(ast.NodeTransformer, BaseRewriteGen):
    """
    Base rewrite class
    - name_mapping: dict[str, str], map name to another name
    - attr_mapping: dict[str, ast.expr], map attribute to an ast expression
    - attr_rn_fn: Callable[[str, str], str], attribute renaming function
    """

    def __init__(self) -> None:
        super().__init__()
        self._name_mapping = None
        self._attr_mapping = None
        self._attr_rn_fn = None

    @property
    def name_mapping(self) -> dict[str, str]:
        """mapping between names"""
        return self._name_mapping

    @name_mapping.setter
    def name_mapping(self, val: dict[str, str]) -> None:
        self._name_mapping = val

    @property
    def attr_mapping(self) -> dict[str, ast.expr]:
        """map attribute string to an ast node"""
        return self._attr_mapping

    @attr_mapping.setter
    def attr_mapping(self, val: dict[str, ast.expr]) -> None:
        self._attr_mapping = val

    def visit_Name(self, node: ast.Name):
        """rewrite the name in the ast with name_mapping"""
        type_name, ctx = node.id, node.ctx
        return ast.Name(id=self.name_mapping[type_name], ctx=ctx)

    def visit_Attribute(self, node: ast.Attribute):
        """rewrite the attribute in the ast with attr_mapping"""
        value, attr, ctx = node.value, node.attr, node.ctx
        id = self.visit_Name(value).id
        new_name = self._attr_rn_fn(id, attr)
        ast_node = ast.Name(id=new_name, ctx=ctx)
        return ast_node


class SympyRewriteGen(BaseRewriteGen):
    def __init__(self) -> None:
        super().__init__()
        self._mapping = None

    @property
    def mapping(self) -> dict[sympy.Symbol, sympy.Symbol]:
        """mapping between names"""
        return self._mapping

    @mapping.setter
    def mapping(self, val: dict[sympy.Symbol, sympy.Symbol]) -> None:
        self._mapping = val

    def visit(self, expr: sympy.Expr):
        symbols = expr.atoms(sympy.Symbol)
        functions = expr.atoms(sympy.Function)

        sympy_mapping = set()
        for sym in symbols:
            sym_name = sym.name
            if sym_name in self.mapping:
                sympy_mapping.add((sym, sympy.Symbol(self.mapping[sym_name])))
            else:  # not a name then it is an attribute
                [ele_name, attr_name] = sym_name.split(".")
                new_sym_name = self._attr_rn_fn(self.mapping[ele_name], attr_name)
                sympy_mapping.add((sym, sympy.Symbol(new_sym_name)))
        for fn in functions:
            fn_name = fn.name
            [ele_name, attr_name] = fn_name.split(".")
            new_fn_name = self._attr_rn_fn(self.mapping[ele_name], attr_name)
            sympy_mapping.add((sympy.Function(fn_name), sympy.Function(new_fn_name)))
        return expr.subs(list(sympy_mapping))


class VectorizeRewriteGen(ast.NodeTransformer, BaseRewriteGen):
    """Rewrite class for vectorized expressions

    x.a(y * z.b) -> x_a((Ty @ var) * (Tzb @ args))
    - x.a: function attribute. Transformed with the renaming function. TODO: add guard for the case where
    the function attribute is not homogeneous.
    - y: state variable. Transformed with one-hot vectors multiplied by the state variables.
    - z.b: numerical attribute. Transformed with one-hot vectors multiplied by the numerical arguments.

    """

    def __init__(self) -> None:
        super().__init__()
        self._fn_mapping = None
        self._keyword_name_mapping = None
        self._src_attr_mapping, self._dst_attr_mapping = None, None
        self._edge_attr_mapping = None

    def set_mappings(self, **kwargs) -> None:
        """Set the mappings for the rewrite"""
        self._fn_mapping = kwargs["fn_mapping"]
        self._keyword_name_mapping = kwargs["keyword_name_mapping"]
        self._src_attr_mapping, self._dst_attr_mapping = (
            kwargs["src_attr_mapping"],
            kwargs["dst_attr_mapping"],
        )
        self._edge_attr_mapping = kwargs["edge_attr_mapping"]

    def visit_Name(self, node: ast.Name):
        """rewrite the name in the ast with name_mapping"""
        type_name, ctx = node.id, node.ctx
        # Here the type name should only be statefule elemnets, i.e., SRC, DST, or SELF
        # TODO: add a guard for this
        return set_ctx(expr=self._keyword_name_mapping[type_name], ctx=ctx)

    def visit_Attribute(self, node: ast.Attribute):
        """rewrite the attribute in the ast with attr_mapping"""
        value, attr, ctx = node.value, node.attr, node.ctx
        assert isinstance(
            value, ast.Name
        ), "Only `Name`nodes are allowed in the value of an Attribute node."
        f"Got type({value}) = {type(value)} instead."
        id = value.id
        if id == kw_name(SRC) or id == kw_name(SELF):
            attr_mapping = self._src_attr_mapping
        elif id == kw_name(DST):
            attr_mapping = self._dst_attr_mapping
        elif id == kw_name(EDGE):
            attr_mapping = self._edge_attr_mapping
        else:
            raise ValueError(
                f"Invalid attribute node: {node}. Expected 1=SRC, DST, or SELF for the value."
                f"Got {id} instead."
            )
        new_node = set_ctx(expr=attr_mapping[attr], ctx=ctx)
        return new_node

    def visit_Call(self, node: ast.Call):
        # Change the function name to the mapped function name
        fn = node.func
        assert isinstance(
            fn, ast.Attribute
        ), "Expected function to be an attribute node."
        f"Got type({fn}) = {type(fn)} instead."
        value, attr, ctx = fn.value.id, fn.attr, fn.ctx
        new_name = ast.Name(id=self._attr_rn_fn(self._fn_mapping[value], attr), ctx=ctx)
        return ast.Call(
            func=new_name,
            args=[self.visit(arg) for arg in node.args],
            keywords=node.keywords,
        )
