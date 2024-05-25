from abc import ABC, abstractmethod
from typing import Callable

from ark.specification.range import Range


class AttrType(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def val_str(self, val) -> str:
        pass

    @abstractmethod
    def check_valid(self, val) -> bool:
        pass

    @property
    @abstractmethod
    def default(self):
        pass


class AnalogAttr(AttrType):

    def __init__(self, val_range: Range | tuple[float, float] = None):
        if not isinstance(val_range, Range) and val_range is not None:
            val_range = Range(*val_range)
        self.val_range = val_range

    def val_str(self, val: float | int) -> str:
        return str(val)

    def check_valid(self, val) -> bool:
        if not isinstance(val, (float, int)):
            return False
        if self.has_range:
            return self.val_range.check_in_range(val)
        return True

    @property
    def default(self) -> float | int:
        if self.has_range:
            return self.val_range.mean
        else:
            raise ValueError(
                "Cannot get default value for an attribute without valid range"
            )

    @property
    def has_range(self) -> bool:
        return self.val_range is not None


class DigitalAttr(AttrType):

    def __init__(self, val_choices: list):
        assert len(val_choices) > 0
        self.val_choices = val_choices

    def val_str(self, val: int) -> str:
        return str(val)

    def check_valid(self, val) -> bool:
        if val not in self.val_choices:
            return False
        return True

    @property
    def default(self) -> float | int:
        return self.val_choices[0]

    @property
    def n_choices(self) -> int:
        return len(self.val_choices)


class FunctionAttr(AttrType):

    def __init__(self, nargs: int):
        self.nargs = nargs

    def val_str(self, val: Callable) -> str:
        return val.__name__

    def check_valid(self, val) -> bool:
        if not isinstance(val, Callable):
            return False
        if len(val.__code__.co_varnames) != self.nargs:
            return False
        return True

    @property
    def default(self) -> Callable:
        def empty_func(*args):
            raise RuntimeError("Default function not implemented")

        return empty_func
