"""Global variables and constants for the ark package."""
from enum import Enum
from dataclasses import dataclass

class Direction(Enum):
    """Class for keeping track of the direction of an edge in a CDG."""
    IN: int = 0
    OUT: int = 1
    SELF: int = 2

@dataclass
class Range:
    """Class for keeping track of the valid range of a node or edge attribute."""
    min: float = None
    max: float = None

class Attr:
    """Ａttribute definition for a CDGType."""
    def __init__(self, attr_type: type, attr_range: Range=None):
        self.type = attr_type
        self.valid_range = attr_range

    def __repr__(self) -> str:
        return f'Attr(type={self.type}, valid_range={self.valid_range})'

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
    def __init__(self, attr_type: type, attr_range: Range, rstd: float):
        self.rstd = rstd
        super().__init__(attr_type, attr_range)
