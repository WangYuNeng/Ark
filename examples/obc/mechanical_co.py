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

co_spec = CDGSpec("co")

#### Type definitions start ####
# Phase of an oscillator:
# lock_fn: injection locking, e.g., omeag_s sin (2 phi)
# osc_fn: coupling, e.g., omega_c sin (phi_i - phi_j)
Osc = NodeType(name="Osc", order=2, attr_def=[AttrDef("mass", attr_type=float)])

# k: coupling strength
Coupling = EdgeType(name="Coupling", attr_def=[AttrDef("k", attr_type=float)])

cdg_types = [Osc, Coupling]
co_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, -EDGE.k * (VAR(SRC) - VAR(DST)) / SRC.mass)
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, -EDGE.k * (VAR(DST) - VAR(SRC)) / DST.mass)
production_rules = [r_cp_src, r_cp_dst]
co_spec.add_production_rules(production_rules)
#### Production rules end ####

#### Validation rules start ####
#### Validation rules end ####


if __name__ == "__main__":
    system = Ark(cdg_spec=co_spec)
    node1 = Osc(mass=1.0)
    node2 = Osc(mass=1.0)
    cpl = Coupling(k=2.0)

    graph = CDG()
    graph.connect(cpl, node1, node2)

    system.compile(cdg=graph, import_lib={})
    time_range = [0, 10]
    time_points = np.linspace(*time_range, 1000)
    # mapping = compiler.var_mapping
    node1.set_init_val(val=0, n=0)
    node1.set_init_val(val=0.0, n=1)
    node2.set_init_val(val=1, n=0)
    node2.set_init_val(val=0.0, n=1)
    system.execute(cdg=graph, time_eval=time_points)
    for node in [node1, node2]:
        phi = node.get_trace(n=0)
        plt.plot(time_points, phi)
    plt.show()
