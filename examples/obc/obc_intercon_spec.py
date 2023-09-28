"""
Example: Coupled Oscillator Network with Possible Interconnects Options
"""

from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, TIME, VAR
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValPattern, ValRule

N_GROUP = 2
obc_intercon_spec = CDGSpec(name="obc-intercon")

#### Type definitions start ####
Osc = NodeType(name="Osc", order=1)
Coupling = EdgeType(
    name="Cpl",
    attr_def=[AttrDef("k", attr_type=float, attr_range=Range(min=-8, max=8))],
)

Osc_group = [NodeType(name=f"Osc_G{i}", base=Osc) for i in range(N_GROUP)]
# local connection that has a lower cost
Coupling_local = EdgeType(
    name="Cpl_l",
    base=Coupling,
    attr_def=[AttrDef("cost", attr_type=int, attr_range=Range(exact=1))],
)
# global connection that has a higher cost
Coupling_global = EdgeType(
    name="Cpl_g",
    base=Coupling,
    attr_def=[AttrDef("cost", attr_type=int, attr_range=Range(exact=10))],
)
cdg_types = [Osc, Coupling] + Osc_group + [Coupling_local, Coupling_global]
obc_intercon_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, -EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST)))
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, -EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC)))
r_lock = ProdRule(Coupling, Osc, Osc, SELF, -SRC.lock_fn(TIME, VAR(SRC)))
production_rules = [r_cp_src, r_cp_dst, r_lock]
obc_intercon_spec.add_production_rules(production_rules)
#### Production rules end ####

#### Validation rules start ####
val_rules = []
for i in range(N_GROUP):
    node_type = Osc_group[i]
    val_rules.append(
        ValRule(
            node_type,
            [
                ValPattern(SELF, Coupling_local, node_type, Range(exact=1)),
                ValPattern(SRC, Coupling_local, node_type, Range(min=0)),
                ValPattern(DST, Coupling_local, node_type, Range(min=0)),
                ValPattern(SRC, Coupling_global, Osc, Range(min=0)),
                ValPattern(DST, Coupling_global, Osc, Range(min=0)),
            ],
        )
    )
obc_intercon_spec.add_validation_rules(val_rules)
#### Validation rules end ####

if __name__ == "__main__":
    import ark.visualize.latex_gen as latexlib

    latexlib.language_to_latex(obc_intercon_spec)
