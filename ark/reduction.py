"""
Reduction functions for the dynamical system.
"""
from typing import Any


class Reduction:
    """
    Reduction base class.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

class Sum(Reduction):
    """
    Summation
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class Product(Reduction):
    """
    Product
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

SUM = Sum()
PRODUCT = Product()
