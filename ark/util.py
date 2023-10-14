from functools import partial
from typing import Any, Callable


def inspect_func_name(func: Callable[..., Any]) -> str:
    """Return the name of the function.

    Args:
        func (Callable[..., Any]): The function to inspect.

    Returns:
        str: The name of the function.
    """
    if isinstance(func, partial):
        return func.func.__name__
    return func.__name__
