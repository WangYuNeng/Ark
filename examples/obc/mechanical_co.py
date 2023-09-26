"""
Coupled oscillator governed by the Law of Motion
"""
import numpy as np
import matplotlib.pyplot as plt
from ark.specification.attribute_def import AttrDef
from ark.specification.specification import CDGSpec
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, EDGE, VAR
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.cdg.cdg import CDG

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
    compiler = ArkCompiler(rewrite=RewriteGen())
    node1 = Osc(mass=1.0)
    node2 = Osc(mass=2.0)
    cpl = Coupling(k=2.0)

    graph = CDG()
    graph.connect(cpl, node1, node2)

    compiler.compile(cdg=graph, cdg_spec=co_spec, help_fn=[], import_lib={})
    compiler.print_prog()
    time_range = [0, 10]
    time_points = np.linspace(*time_range, 1000)
    mapping = compiler.var_mapping
    init_states = compiler.map_init_state(
        {node: np.random.rand() * 10 for node in mapping.keys()}
    )
    print(init_states)
    sol = compiler.prog(time_range, init_states=init_states, dense_output=True)
    for node, idx in mapping.items():
        phi = sol.sol(time_points)[idx].T
        plt.plot(time_points, phi)
    plt.show()
