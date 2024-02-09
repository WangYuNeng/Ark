from importlib.util import find_spec
_required_packages = ['jax', 'pint', 'equinox', 'diffrax']
_missing_packages = [
    package for package in _required_packages if find_spec(package) is None
]
if _missing_packages:
    raise ImportError(
        f"To use the differentiable simulation part of Ark, please install the following packages: {', '.join(_missing_packages)}"
    )