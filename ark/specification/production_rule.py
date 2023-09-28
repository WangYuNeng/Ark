"""
Ark production Rule
"""
import ast
from dataclasses import dataclass

import sympy

from ark.cdg.cdg import CDGEdge, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.rule_keyword import (
    DST,
    EDGE,
    SELF,
    SRC,
    TIME,
    Expression,
    Target,
    kw_name,
)


@dataclass
class ProdRuleId:
    """Production Rule Identifier Class"""

    et: EdgeType
    src_nt: NodeType
    dst_nt: NodeType
    gen_tgt: Target

    def __hash__(self) -> int:
        return repr(
            [self.et.name, self.src_nt.name, self.dst_nt.name, kw_name(self.gen_tgt)]
        ).__hash__()

    def __str__(self) -> str:
        return str(
            [self.et.name, self.src_nt.name, self.dst_nt.name, kw_name(self.gen_tgt)]
        )


class ProdRule:
    """Production Rule Class"""

    def __init__(
        self,
        et: EdgeType,
        src_nt: NodeType,
        dst_nt: NodeType,
        gen_tgt: Target,
        fn_exp: Expression,
    ) -> None:
        self._et = et
        self._src_nt = src_nt
        self._dst_nt = dst_nt
        self._gen_tgt = gen_tgt
        self._id = ProdRuleId(et, src_nt, dst_nt, gen_tgt)
        self._fn_exp = fn_exp

    @staticmethod
    def get_identifier(
        et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: Target
    ) -> ProdRuleId:
        """Returns a unique identifier for the production rule"""
        return ProdRuleId(et.name, src_nt.name, dst_nt.name, kw_name(gen_tgt))

    @property
    def identifier(self) -> ProdRuleId:
        """Unique identifier for the production rule"""
        return self._id

    @property
    def fn_ast(self) -> ast.Expr:
        """Returns the AST of the production function
        TODO: Change to a more pythonic way of doing this, e.g., overload
        the arithmetic operators.
        """
        return ast.parse(str(self._fn_exp), mode="eval")

    @property
    def fn_sympy(self) -> sympy.Expr:
        """Returns the sympy expression of the production function"""
        if type(self._fn_exp) is int:
            return sympy.Float(self._fn_exp)
        else:
            return self._fn_exp.sympy

    def get_rewrite_mapping(self, edge: CDGEdge):
        """
        Returns a dictionary that maps the keyword in production rules to the name
        of the nodes and edges in the CDG.
        """
        src: CDGNode
        dst: CDGNode

        src, dst = edge.src, edge.dst
        name_map = {
            kw_name(EDGE): edge.name,
            kw_name(SRC): src.name,
            kw_name(DST): dst.name,
            kw_name(SELF): src.name,
            kw_name(TIME): kw_name(TIME),
        }
        return name_map

    def __str__(self) -> str:
        return f"{self.identifier}: {self._fn_exp}"
