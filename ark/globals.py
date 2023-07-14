"""Global variables and constants for the ark package."""
from enum import Enum

class Direction(Enum):
    """Class for keeping track of the direction of an edge in a CDG."""
    IN: int = 0
    OUT: int = 1
    SELF: int = 2
