"""
Attribute class for CDGType.
"""
from typing import NewType, Union
from types import FunctionType
from ark.specification.range import Range

AttrImpl = NewType('AttrImpl', Union[int, float, FunctionType]) # why pylint error?


class AttrDef:
    """Ａttribute Definition for a CDGType."""
    def __init__(self, name: str, attr_type: type, attr_range: Range=None):
        self.name = name
        self.type = attr_type
        self.valid_range = attr_range

    def __repr__(self) -> str:
        return f'AttrDef(name={self.name}, type={self.type}, \
            valid_range={self.valid_range})'

    def attr_str(self, val: AttrImpl) -> str:
        """Get an AST expression for this attribute."""
        if self.type == int or self.type == float:
            val_str = str(val)
        elif self.type == FunctionType:
            val_str = val.__name__
        else:
            raise NotImplementedError(f'AST expression for type {self.type} not implemented')
        return val_str

    def check(self, val: AttrImpl) -> bool:
        """Check if val is in the valid range of this attribute."""
        if not isinstance(val, self.type):
            raise TypeError(f'Expected type {self.type}, got {type(val)}')
        if self.valid_range is None:
            return True
        if not self.valid_range.check_in_range(val):
            raise ValueError(f'Expected value in range {self.valid_range}, got {val}')
        return True

class AttrDefMismatch(AttrDef):
    """Attribute definition for a CDGType where the value is sampled 
    from a normal distribution to model the random mismatch in hardware.
    
    Args:
        rstd: relative standard deviation of the random value 
        
    The check() method only check whether the nominal value is in range
    and does not check the random value.
    """
    def __init__(self, name: str, attr_type: type, attr_range: Range, rstd: float):
        self.rstd = rstd
        super().__init__(name, attr_type, attr_range)

    def attr_str(self, val: AttrImpl) -> str:
        if not self.type == float:
            raise NotImplementedError(f'AST expression for a mismatched attribute \
                                      should be float, not {self.type}')
        return f'np.random.normal({val}, {val} * {self.rstd})'
