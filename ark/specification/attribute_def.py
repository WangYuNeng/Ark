"""
Attribute class for CDGType.
"""

from typing import Callable, NewType, Optional, Union

import numpy as np

from ark.specification.attribute_type import AttrType, FunctionAttr, Trainable

AttrImpl = NewType(
    "AttrImpl", Union[int, float, Callable, Trainable]
)  # why pylint error?


class AttrDef:
    """Ａttribute Definition for a CDGType."""

    def __init__(
        self,
        attr_type: AttrType,
    ):
        self.attr_type = attr_type

    def check(self, val) -> bool:
        """Check if val is in the valid range of this attribute."""

        # Special case: partial function is also consdiered as a function type.
        return self.attr_type.check_valid(val)

    @property
    def default(self) -> AttrImpl:
        """Returen the default value of this attribute.

        If attr_type is a function, return an empty function that raises error.
        For attr_type being int or float, return a value in the range.

        Returns:
            AttrImpl: The default value of this attribute.
        """

        return self.attr_type.default


class AttrDefMismatch(AttrDef):
    """Attribute definition for a CDGType where the value is sampled
    from a normal distribution to model the random mismatch in hardware.

    Args:
        rstd: relative standard deviation of the random value

    The check() method only check whether the nominal value is in range
    and does not check the random value.
    """

    def __init__(
        self,
        attr_type: AttrDef,
        rstd: Optional[float] = None,
        std: Optional[float] = None,
    ):
        if (rstd or std) and isinstance(attr_type, FunctionAttr):
            raise ValueError("Function attribute don't have mismatch")
        if rstd and std:
            raise ValueError("Cannot specify both rstd and std")
        if not rstd and not std:
            raise ValueError("Must specify either rstd or std")
        self.rstd = rstd
        self.std = std
        super().__init__(attr_type)

    def sample(self, mean: float) -> float:
        """Sample a random value from the normal distribution.

        Args:
            mean (float): mean of the normal distribution
        Returns:
            float: a random value from the normal distribution
        """
        if self.rstd:
            return np.random.normal(mean, np.abs(mean * self.rstd))
        else:
            return np.random.normal(mean, self.std)
            return np.random.normal(mean, self.std)
