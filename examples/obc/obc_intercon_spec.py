"""
Example: Coupled Oscillator Network with Possible Interconnects Options
"""

from ark.ark import Ark
from ark.cdg.cdg import CDG
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
Osc = NodeType(name="Osc", attrs={"order": 1})
Coupling = EdgeType(
    name="Cpl",
    attrs={"attr_def": {"k": AttrDef(attr_type=float, attr_range=Range(min=0.0))}},
)

Osc_group = [NodeType(name=f"Osc_G{i}", bases=Osc) for i in range(N_GROUP)]
# local connection that has a lower cost
Coupling_local = EdgeType(
    name="Cpl_l",
    bases=Coupling,
    attrs={"attr_def": {"cost": AttrDef(attr_type=int, attr_range=Range(exact=1))}},
)
# global connection that has a higher cost
Coupling_global = EdgeType(
    name="Cpl_g",
    bases=Coupling,
    attrs={"attr_def": {"cost": AttrDef(attr_type=int, attr_range=Range(exact=10))}},
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
    valid_graph, invalid_graph = CDG(), CDG()

    # Create a valid graph - connecting oscillators outside groups
    # with global connection
    osc0, osc1 = Osc_group[0](), Osc_group[1]()
    global_cp = Coupling_global(k=1.0, cost=10)
    self_locking0 = Coupling_local(k=1.0, cost=1)
    self_locking1 = Coupling_local(k=1.0, cost=1)
    valid_graph.connect(global_cp, osc0, osc1)
    valid_graph.connect(self_locking0, osc0, osc0)
    valid_graph.connect(self_locking1, osc1, osc1)

    # Create an invalid graph - connecting oscillators outside groups
    # with local connection
    osc2, osc3 = Osc_group[0](), Osc_group[1]()
    local_cp = Coupling_local(k=1.0, cost=1)
    self_locking2 = Coupling_local(k=1.0, cost=1)
    self_locking3 = Coupling_local(k=1.0, cost=1)
    invalid_graph.connect(local_cp, osc2, osc3)
    invalid_graph.connect(self_locking2, osc2, osc2)
    invalid_graph.connect(self_locking3, osc3, osc3)

    # Validate the graphs
    system = Ark(cdg_spec=obc_intercon_spec)
    print("Validating oscillators in different groups coupled with global connection")
    system.validate(valid_graph)  # pass
    print("PASSED!\n")
    print("Validating oscillators in different groups coupled with local connection")
    system.validate(invalid_graph)  # fail, raise erroe
    print("PASSED!\n")
