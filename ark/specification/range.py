from dataclasses import dataclass

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
