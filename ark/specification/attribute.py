"""
Attribute class for CDGType.
"""
from dataclasses import dataclass
from typing import Any
from types import FunctionType
import ast


@dataclass
class Range:
    """Class for keeping track of the valid range of a node or edge attribute."""
    min: float = None
    max: float = None

class Attr:
    """Ａttribute for a CDGType."""
    def __init__(self, name: str, attr_type: type, attr_range: Range):
        self.name = name
        self.type = attr_type
        self.valid_range = attr_range
        self.value = None

    def __repr__(self) -> str:
        return f'Attr(name={self.name}, type={self.type}, \
            valid_range={self.valid_range}, value={self.value})'

    def set_val(self, val) -> None:
        """Set the value of this attribute."""
        if self.check(val):
            self.value = val

    def get_val(self) -> Any:
        """Get the value of this attribute."""
        return self.value

    def ast_expr(self) -> ast.Expr:
        """Get an AST expression for this attribute."""
        if self.type == int or self.type == float:
            val = str(self.value)
        elif self.type == FunctionType:
            val = self.value.__name__
        else:
            raise NotImplementedError(f'AST expression for type {self.type} not implemented')
        mod = ast.parse(val)
        return mod.body[0].value

    def check(self, val) -> bool:
        """Check if val is in the valid range of this attribute."""
        if not isinstance(val, self.type):
            raise TypeError(f'Expected type {self.type}, got {type(val)}')
        if self.valid_range is None:
            return True
        if self.valid_range.min is not None and val < self.valid_range.min or \
            self.valid_range.max is not None and val > self.valid_range.max:
            raise ValueError(f'Expected value in range {self.valid_range}, got {val}')
        return True

class MismatchAttr(Attr):
    """Attribute definition that will for a CDGType."""
    def __init__(self, name: str, attr_type: type, attr_range: Range, rstd: float):
        self.rstd = rstd
        super().__init__(name, attr_type, attr_range)

    def ast_expr(self) -> ast.Expr:
        if not self.type == float:
            raise NotImplementedError(f'AST expression for a mismatched attribute \
                                      should be float, not {self.type}')
        val = f'np.random.normal({self.value}, {self.rstd})'
        return ast.parse(val).body[0].value
