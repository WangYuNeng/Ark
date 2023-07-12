"""
Ark Generation Rule
"""
import ast
from typing import List
from dataclasses import dataclass

class Expression:
    """Expression for writing generation rules
    Ref: https://github.com/ncsys-lab/analog-verification/blob/new-modelspec/core/expr.py
    """

    def __add__(self, other: "Expression | float | int") -> "Sum":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Sum(self, other)

    def __sub__(self, other: "Expression | float | int") -> "Difference":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Difference(self, other)

    def __mul__(self, other: "Expression | float | int") -> "Product":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Product(self, other)

    def __truediv__(self, other: "Expression | float | int") -> "Quotient":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Quotient(self, other)

    def __neg__(self) -> "Negation":
        return Negation(self)

    def __pow__(self, other: "Expression | float | int") -> "Power":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Power(self, other)

    def __call__(self, *args: "List[Expression | float | int]") -> "FunctionCall":
        converted_args = []
        for arg in args:
            if isinstance(arg, int):
                arg = Constant(float(arg))
            if isinstance(arg, float):
                arg = Constant(arg)
            converted_args.append(arg)
        return FunctionCall(self, converted_args)

@dataclass
class Variable(Expression):
    """Variable expression"""
    name: str

    def __str__(self) -> str:
        return self.name

@dataclass
class Constant(Expression):
    """Constant expression"""
    value: float

    def __str__(self) -> str:
        return str(self.value)

@dataclass
class Sum(Expression):
    """Sum expression"""
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f'({self.left} + {self.right})'

@dataclass
class Difference(Expression):
    """Difference expression"""
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f'({self.left} - {self.right})'

@dataclass
class Product(Expression):
    """Product expression"""
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f'({self.left} * {self.right})'

@dataclass
class Quotient(Expression):
    """Quotient expression"""
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f'({self.left} / {self.right})'

@dataclass
class Negation(Expression):
    """Negation expression"""
    expr: Expression

    def __str__(self) -> str:
        return f'(-{self.expr})'

@dataclass
class Power(Expression):
    """Power expression"""
    base: Expression
    exp: Expression

    def __str__(self) -> str:
        return f'({self.base} ** {self.exp})'

@dataclass
class FunctionCall(Expression):
    """Function call expression"""
    fn: Expression
    args: List[Expression]

    def __str__(self) -> str:
        return f'{self.fn}({", ".join(str(arg) for arg in self.args)})'

@dataclass
class GenRuleKeyword:
    """Keyword class for writing generation rules"""
    name: str

    def __getattribute__(self, __name: str) -> Variable:
        if __name.startswith('__') and __name.endswith('__'):
            # Handle python default attributes
            return object.__getattribute__(self, __name)
        else:
            self_name = object.__getattribute__(self, 'name')
            named_attr = f'{self_name}.{__name}'
            return Variable(named_attr)

    def __repr__(self) -> str:
        return self.name

SRC, DST = GenRuleKeyword('SRC'), GenRuleKeyword('DST'),
EDGE, SELF = GenRuleKeyword('EDGE'), GenRuleKeyword('SELF')
TIME = Variable('TIME')

def VAR(gen_rule_keyword: GenRuleKeyword) -> Variable:
    """Returns a state variable for writing generation rules"""
    name = object.__getattribute__(gen_rule_keyword, 'name')
    return Variable(name)

def kw_name(keyword: GenRuleKeyword):
    """Returns the name of the keyword"""
    return object.__getattribute__(keyword, 'name')

class GenRule:
    """Generation Rule Class"""

    def __init__(self, tgt_et: "EdgeType", src_nt: "NodeType", dst_nt: "NodeType",
                 gen_tgt: GenRuleKeyword, fn_exp: Expression) -> None:
        self._tgt_et = tgt_et
        self._src_nt = src_nt
        self._dst_nt = dst_nt
        self._gen_tgt = gen_tgt
        self._fn_ast = ast.parse(str(fn_exp), mode='eval')

    @staticmethod
    def get_identifier(tgt_et: "EdgeType", src_nt: "NodeType", dst_nt: "NodeType",
                       gen_tgt: GenRuleKeyword):
        """Returns a unique identifier for the generation rule"""
        return repr([tgt_et.name, src_nt.name, dst_nt.name, kw_name(gen_tgt)])

    @property
    def identifier(self):
        """Unique identifier for the generation rule"""
        return self.get_identifier(self._tgt_et, self._src_nt, self._dst_nt, self._gen_tgt)

    @property
    def fn_ast(self):
        """Returns the AST of the generation function 
        TODO: Change to a more pythonic way of doing this, e.g., overload
        the arithmetic operators.
        """
        return self._fn_ast

    def get_rewrite_mapping(self, edge: 'CDGEdge'):
        """
        Returns a dictionary that maps the keyword in generation rules to the name
        of the nodes and edges in the CDG.
        """
        src: 'CDGNode'
        dst: 'CDGNode'

        src, dst = edge.src, edge.dst
        name_map = {kw_name(EDGE): edge.name, kw_name(SRC): src.name, kw_name(DST): dst.name}
        return name_map
