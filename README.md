# ARK: A Programming Language for Agile Development of Unconventional Computing Paradigms

This package implements the language described in [Design of Novel Analog Compute Paradigms with Ark](https://arxiv.org/abs/2309.08774). 

## Build from source
To setup the environment, please have `graphviz`, and a SMT solver in the system and run

`pip install .` (or `pip install -e .` if you'd like to edit the package).

Additional configuration might be needed to set up the binding between `pySMT` and the solver[^1].

The package is tested with `python 3.10.13` and `python3.11.3`.

### Additional Dependencies

If you'd like to run random_tln.py, please also have `python3.10` (using >3.10 will have dependecy issue) and install ngspice and pyspice in the system.

```
apt-get install libngspice0
conda install -c conda-forge pyspice
```

## Run in a container 
Note: This is not tested in the current version. If you want to run this please refer to [this commit](https://github.com/WangYuNeng/Ark/tree/baa94b989f4e6064d122d99ac72961905faeadc8)
```
./build_image.sh
./run_image.sh
```

## Quick Start

To generate the figures and run the experiments in the paper, run

```
./scripts/run_exp.sh
```

The results will be stored in `output` directory and also shwon in the terminal.

Here's a very simple example to describe mechanical coupled oscillator governed by the Law of Motion with Ark

```python
"""
Coupled oscillator governed by the Law of Motion
"""
from ark.ark import Ark
from ark.cdg.cdg import CDG
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SRC, VAR
from ark.specification.specification import CDGSpec
from ark.specification.attribute_type import AnalogAttr

# Specification of coupled oscillator
co_spec = CDGSpec("co")

#### Type definitions start ####
# Oscillator node: The state variable models the displacement of the oscillator
# Order = 2 means the state variable controlled by its second derivative - acceleration
# The mass attribute models the mass of the oscillator
Osc = NodeType(
    name="Osc",
    attrs={"order": 2, "attr_def": {"mass": AttrDef(attr_type=AnalogAttr((0, 10)))}},
)

# Coupling springs
# k: coupling strength
Coupling = EdgeType(
    name="Coupling", attrs={"attr_def": {"k": AttrDef(attr_type=AnalogAttr((0, 10)))}}
)

cdg_types = [Osc, Coupling]
co_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
# F = ma = -kx -> a = x'' = -kx/m
r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, -EDGE.k * (VAR(SRC) - VAR(DST)) / SRC.mass)
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, -EDGE.k * (VAR(DST) - VAR(SRC)) / DST.mass)
production_rules = [r_cp_src, r_cp_dst]
co_spec.add_production_rules(production_rules)
#### Production rules end ####

#### Validation rules start ####
#### Validation rules end ####
```

Ideally, hardware designers write down the **specification** (or the **language** using the formalization from the paper). The specification contains type definition (`cdg_types`), production rules (`production_rules`), and validation rules (omitted here). Domain specialists can design applications using the specification.

```python
import matplotlib.pyplot as plt
import numpy as np

# Manipulate the dynamical system with the CDG and execute it with Ark
system = Ark(cdg_spec=co_spec)

# Initialize 2 couplied oscillators
graph = CDG()
node1 = Osc(mass=1.0)
node2 = Osc(mass=2.0)
cpl = Coupling(k=2.0)
graph.connect(cpl, node1, node2)

# Compile the CDG to an executable dynamical system
system.compile(cdg=graph)

# Comment out this line if you want this to run in a python notebook
system.dump_prog("../../output/mechanical_co-prog.py")

# Specify the simulation time and initial values
time_range = [0, 10]
time_points = np.linspace(*time_range, 1000)
node1.set_init_val(val=0, n=0)
node1.set_init_val(val=0.0, n=1)
node2.set_init_val(val=1, n=0)
node2.set_init_val(val=0.0, n=1)

# Execute the dynamical system
system.execute(cdg=graph, time_eval=time_points)

# Retrieve the traces of the state variables for the CDG
for node in [node1, node2]:
    phi = node.get_trace(n=0)
    plt.plot(time_points, phi, label=node.name)
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend()

# Comment out this line if you want this to run in a python notebook
plt.savefig("../../output/mechanical_co-transient.pdf")
```

The resulting figures show the oscillating behaviors.
![Dynamics of the two oscillators](https://github.com/WangYuNeng/Ark/tree/main/examples/obc/co.png)

## More Examples

You can try more examples inside the `examples` directory.

- `cnn`: A specification for the cellular nonlinear network and an edge-detection application.
- `obc`: A specification for the oscillator-based-computing paradigm and a max-cut application.
- `tln`: A specification for the transmission-line-network and examaples demonstrating how mismatches affects the system response.
- `crn`: A toy example that models the chemical reaction network with Ark and implements an multiplication.
- `n_path_filter`: A toy example that models the n-path filter with Ark.
For more details, please refer to the descriptions in the paper.


[^1]: For example, in macOS, it is possible that z3 solver failed to find the `libz3.dylib` even when z3 is installed with brew. This because somehow z3 looks for librariesin `/op/how/bin/` instead of `/opt/homebrew/lib`. One workaround is copying the `libz3.dylib` in`/opt/homebrew/lib` to a directory that z3 will access, e.g., `/Users/{username}/miniconda3/envs/{envname}/lib`. For more information, please refer to <https://github.com/pysmt/pysmt>
