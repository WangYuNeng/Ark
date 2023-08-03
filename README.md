# ARK: A Design System for Agile Development of Unconventional Computing Paradigms
To setup the environment, please have `ngspice`, `graphviz`, and an SMT solver in the system and run 
    `pip install -e .` in the `Ark` directory.
You might need additional configuration to set up the binding between `pySMT` and the solver.
For more information, please refer to https://github.com/pysmt/pysmt

## Examples
    - tests/rlc.py

Legacy saa codebase (same idea but more ad-hoc implementation)
    - A ladder filter
    - Beamformer with diffrent input angle
