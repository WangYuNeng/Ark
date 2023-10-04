"""
Example: Chemical Reaction Network
https://link.springer.com/article/10.1007/s11047-019-09775-1

Idea maybe? https://pubs.acs.org/doi/10.1021/acssynbio.0c00050
- CRN++ might result in CRNs that are hard to implement in practice
- ... This highlights a niche for DSLs at medium levels of abstraction
  that are geared specificallyfor modeling computational nucleic acid
  devices such as thosereviewed above.
"""
import matplotlib.pyplot as plt
import numpy as np

from ark.ark import Ark
from ark.cdg.cdg import CDG
from ark.reduction import PRODUCT
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SRC, VAR
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValPattern, ValRule

Cpd = NodeType(name="Cpd", attrs={"order": 1})
Rct = NodeType(
    name="Rct",
    attrs={
        "order": 0,
        "reduction": PRODUCT,
        "attr_def": {"k": AttrDef(attr_type=float, attr_range=Range(min=0.0, max=1.0))},
    },
)
Rct_et = EdgeType(
    name="Rct_et",
    attrs={
        "attr_def": {
            "nc": AttrDef(attr_type=int),
            "coeff": AttrDef(attr_type=int, attr_range=Range(min=1)),
        }
    },
)

rule1 = ProdRule(Rct_et, Cpd, Rct, SRC, DST.k * EDGE.nc * VAR(DST))
rule2 = ProdRule(Rct_et, Rct, Cpd, DST, SRC.k * EDGE.nc * VAR(SRC))
rule3 = ProdRule(Rct_et, Cpd, Rct, DST, VAR(SRC) ** EDGE.coeff)

# Example: Multiplication C = A * B
# Compound A, B, C, D
# Reaction 1: A + B -> A + B + C
# Reaction 2: C -> D
A, B = NodeType(name="A", bases=Cpd), NodeType(name="B", bases=Cpd)
C, D = NodeType(name="C", bases=Cpd), NodeType(name="D", bases=Cpd)
R1, R2 = NodeType(name="R1", bases=Rct), NodeType(name="R2", bases=Rct)

# Validation rules: Reaction must happen if presented
# Problem: The resulting CDG would always represent all possible reactions,
#          just with different parameters.
# Or, need to be specified in custom function to check the existence of other chemical species
a_val = ValRule(A, [ValPattern(SRC, Rct_et, R1, Range(exact=1))])
b_val = ValRule(B, [ValPattern(SRC, Rct_et, R1, Range(exact=1))])
c_val = ValRule(
    C,
    [
        ValPattern(DST, Rct_et, R1, Range(exact=1)),
        ValPattern(SRC, Rct_et, R2, Range(exact=1)),
    ],
)
d_val = ValRule(D, [ValPattern(DST, Rct_et, R2, Range(exact=1))])
r1_val = ValRule(
    R1,
    [
        ValPattern(DST, Rct_et, A, Range(exact=1)),
        ValPattern(DST, Rct_et, B, Range(exact=1)),
        ValPattern(SRC, Rct_et, C, Range(exact=1)),
    ],
)
r2_val = ValRule(
    R2,
    [
        ValPattern(DST, Rct_et, C, Range(exact=1)),
        ValPattern(SRC, Rct_et, D, Range(exact=1)),
    ],
)
val_rules = [a_val, b_val, c_val, d_val, r1_val, r2_val]

a, b, c, d = A(), B(), C(), D()
r1, r2 = R1(k=1.0), R2(k=1.0)
e_1a = Rct_et(nc=0, coeff=1)
e_1b = Rct_et(nc=0, coeff=1)
e_1c = Rct_et(nc=1, coeff=1)

e_2c = Rct_et(nc=-1, coeff=1)
e_2d = Rct_et(nc=1, coeff=1)

graph = CDG()
graph.connect(e_1a, a, r1)
graph.connect(e_1b, b, r1)
graph.connect(e_1c, r1, c)
graph.connect(e_2c, c, r2)
graph.connect(e_2d, r2, d)

cdg_types = [A, B, C, D, R1, R2]
production_rules = [rule1, rule2, rule3]
spec = CDGSpec(
    cdg_types=cdg_types, production_rules=production_rules, validation_rules=val_rules
)
system = Ark(
    cdg_spec=spec,
)
assert system.validate(cdg=graph)
system.compile(cdg=graph)

time_range = [0, 15]
time_points = np.linspace(*time_range, 1000)
a.set_init_val(val=6, n=0)
b.set_init_val(val=2, n=0)
c.set_init_val(val=0, n=0)
d.set_init_val(val=0, n=0)
system.execute(cdg=graph, time_eval=time_points, init_seed=0, sim_seed=0)
for node in [a, b, c]:
    trace = node.get_trace(n=0)
    plt.plot(time_points, trace, label=node.name)
plt.xlabel("time")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.savefig("crn.png")
plt.show()
