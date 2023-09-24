from setuptools import setup

setup(
    name="ark",
    version="0.1.0",
    description="Spatial Analog Abstraction",
    packages=["ark"],
    install_requires=[
        "pyspice",
        "matplotlib",
        "numpy",
        "pysmt",
        "tqdm",
        "sympy",
        "palletable",
        "graphviz",
    ],
)
