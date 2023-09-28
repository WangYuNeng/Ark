from dataclasses import dataclass

from pysmt.shortcuts import GE, LE, And, Equals, Int


@dataclass
class Range:
    """Class for keeping track of the valid range of a node or edge attribute."""

    min: int | float = None
    max: int | float = None
    exact: int | float = None

    def is_interval_bound(self):
        return self.is_range() and not self.min is None and not self.max is None

    def is_lower_bound(self):
        return self.is_range() and self.max is None and not self.min is None

    def is_upper_bound(self):
        return self.is_range() and self.min is None and not self.max is None

    def is_unbounded(self):
        return self.is_range() and self.min is None and self.max is None

    def is_range(self):
        return self.exact is None

    def is_exact(self):
        return not self.exact is None

    def check_in_range(self, val):
        """Check if val is in the valid range."""
        if (
            self.exact is not None
            and val != self.exact
            or self.min is not None
            and val < self.min
            or self.max is not None
            and val > self.max
        ):
            return False
        return True

    def to_str(self, val):
        """Return the string representation of the range."""
        if self.exact is not None:
            return f"{val}={self.exact}"
        if self.min is not None and self.max is not None:
            return f"{self.min}<={val}<={self.max}]"
        if self.min is not None:
            return f"{self.min}<={val}"
        if self.max is not None:
            return f"{val}<={self.max}"
        return "None"

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
