from contextlib import contextmanager
from time import perf_counter
from typing import Literal

import matplotlib as mpl

try:
    from typeguard import typechecked
except ImportError:

    def typechecked(func):
        return func


@typechecked
def matplotlib_settings(
    vector_graphics: bool = True,
    use_latex: bool = False,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 200,
):
    """Set up matplotlib for use in Jupyter notebooks with sensible defaults."""
    import matplotlib_inline

    if vector_graphics:
        matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
    else:
        matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
    mpl.rcParams["text.usetex"] = use_latex
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["figure.dpi"] = dpi


@typechecked
def jax_settings(
    *,
    use_x64: bool,
    device: Literal['cpu', 'gpu'],
    log_compiles: bool,
    num_traceback_frames: int,
) -> dict:
    """
    Set up Jax using subset of settings that are relevant for simulations and debugging them.
    Make sure to run this before importing Jax.

    :param use_x64: Should use 64-bit floating point numbers
    :param device: CPU or GPU
    :param log_compiles: Should log compilation of Jax functions
    :param num_traceback_frames: Number of frames to include in Jax tracebacks
    :return: Jax config values after changing them
    """
    import os

    match device:  # Flags set before Jax initialization
        case "cpu":
            import multiprocessing
            # Set the number of CPU devices to the number of cores
            os.environ[
                "XLA_FLAGS"
            ] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count}"
        case "gpu":
            pass
    import jax

    # Jax configuration settings
    jax.config.update("jax_platform_name", device)
    jax.config.update("jax_threefry_partitionable", True)  # Allows for automatic parallelization when using random numbers
    jax.config.update("jax_enable_x64", use_x64)
    jax.config.update("jax_log_compiles", log_compiles)
    jax.config.update("jax_tracer_error_num_traceback_frames", num_traceback_frames)

    return jax.config.values


@contextmanager
def time_block(name: str):
    try:
        start_time = perf_counter()
        print(f"Starting {name}...")
        yield
    finally:
        end_time = perf_counter()
        print(f"Finished {name} in {end_time - start_time:.3f} seconds.")


def timed_execution(f: callable, *, name: str | None = None, jit_only: bool = False):
    """
    Function wrapper to time execution of a jitted function.
    Note that this turns the function into a blocking one, as it calls `jax.block_until_ready`.
    """
    from functools import wraps
    import jax
    import equinox as eqx

    if name is None:
        name = f.__name__

    timer = None

    def start_timer():
        nonlocal timer
        print(f"Starting process {name}")
        timer = perf_counter()

    def end_timer():
        nonlocal timer
        if timer is None:
            print("Timer initialization failed...")
        total_time = perf_counter() - timer
        print(f"Process {name} took {total_time}")
        timer = None

    def to_jit(*args, **kwargs):
        if jit_only:
            jax.debug.callback(start_timer)
        result = f(*args, **kwargs)
        return result

    jitted = eqx.filter_jit(to_jit)

    @wraps(f)
    def inner(*args, **kwargs):
        if not jit_only:
            start_timer()
        result = jax.block_until_ready(jitted(*args, **kwargs))
        end_timer()
        return result

    return inner
