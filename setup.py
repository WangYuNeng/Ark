from setuptools import setup

setup(
    name = 'saa',
    version = '0.1.0',
    description = 'Spatial Analog Abstraction',
    packages=['saa'],
    install_requires=['pyspice', 'matplotlib', 'numpy']
)