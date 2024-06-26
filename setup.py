from setuptools import find_packages, setup

setup(
    name="ark",
    version="0.1.0",
    description="Spatial Analog Abstraction",
    packages=find_packages(),
    install_requires=[
        "scipy>=1.11.0",
        "matplotlib>=3.7.1",
        "numpy>=1.25.0",
        "pySMT>=0.9.5",
        "tqdm>=4.65.0",
        "sympy>=1.12",
        "palettable>=3.3.3",
        "pylatexenc>=2.10",
        "graphviz>=0.20.1",
        "jax>=0.4.24",
        "optax>=0.1.9",
        "equinox>=0.11.3",
        "jaxtyping>=0.2.25",
        "diffrax>=0.5.0",
    ],
)
