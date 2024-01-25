"""
Coupled oscillator governed by the Law of Motion
"""
import matplotlib.pyplot as plt
import numpy as np

from ark.ark import Ark
from ark.cdg.cdg import CDG
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SRC, VAR
from ark.specification.specification import CDGSpec

# Specification of coupled oscillator
co_spec = CDGSpec("co")

#### Type definitions start ####
# Oscillator node: The state variable models the displacement of the oscillator
# Order = 2 means the state variable controlled by its second derivative - acceleration
# The mass attribute models the mass of the oscillator
Osc = NodeType(
    name="Osc",
    attrs={"order": 2, "attr_def": {"mass": AttrDef(attr_type=float)}},
)

# Coupling springs
# k: coupling strength
Coupling = EdgeType(
    name="Coupling", attrs={"attr_def": {"k": AttrDef(attr_type=float)}}
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
plt.savefig("../../output/mechanical_co-transient.pdf")
