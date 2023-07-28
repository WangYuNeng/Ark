from dataclasses import dataclass
from pysmt.shortcuts import GE, LE, Equals, Int, And

@dataclass
class Range:
    """Class for keeping track of the valid range of a node or edge attribute."""
    min: int | float = None
    max: int | float = None
    exact: int | float = None

    def check_in_range(self, val):
        """Check if val is in the valid range."""
        if self.exact is not None and val != self.exact or \
            self.min is not None and val < self.min or \
            self.max is not None and val > self.max:
            return False
        return True

    def to_str(self, val):
        """Return the string representation of the range."""
        if self.exact is not None:
            return f'{val}={self.exact}'
        if self.min is not None and self.max is not None:
            return f'{self.min}<={val}<={self.max}]'
        if self.min is not None:
            return f'{self.min}<={val}'
        if self.max is not None:
            return f'{val}<={self.max}'
        return 'None'

    def to_smt(self, val):
        """Return the SMT representation of the range."""
        if self.exact is not None:
            return Equals(val, Int(self.exact))
        if self.min is not None and self.max is not None:
            return And(GE(val, Int(self.min)), LE(val, Int(self.max)))
        if self.min is not None:
            return GE(val, Int(self.min))
        if self.max is not None:
            return LE(val, Int(self.max))
        raise NotImplementedError
