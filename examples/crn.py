"""
Example: Chemical Reaction Network
https://link.springer.com/article/10.1007/s11047-019-09775-1

Idea maybe? https://pubs.acs.org/doi/10.1021/acssynbio.0c00050
- CRN++ might result in CRNs that are hard to implement in practice
- ... This highlights a niche for DSLs at medium levels of abstraction
  that are geared specificallyfor modeling computational nucleic acid
  devices such as thosereviewed above.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.attribute_def import AttrDef
from ark.specification.range import Range
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.validation_rule import ValRule, ValPattern
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from ark.reduction import SUM, PRODUCT

Cpd = NodeType(name="Cpd", order=1)
Rct = NodeType(
    name="Rct",
    order=0,
    reduction=PRODUCT,
    attr_def=[AttrDef("k", attr_type=float, attr_range=Range(min=0.0, max=1.0))],
)
Rct_et = EdgeType(
    name="Rct_et",
    attr_def=[
        AttrDef("nc", attr_type=int),
        AttrDef("coeff", attr_type=int, attr_range=Range(min=1)),
    ],
)

rule1 = ProdRule(Rct_et, Cpd, Rct, SRC, DST.k * EDGE.nc * VAR(DST))
rule2 = ProdRule(Rct_et, Rct, Cpd, DST, SRC.k * EDGE.nc * VAR(SRC))
rule3 = ProdRule(Rct_et, Cpd, Rct, DST, VAR(SRC) ** EDGE.coeff)

# Example: Multiplication C = A * B
# Compound A, B, C, D
# Reaction 1: A + B -> A + B + C
# Reaction 2: C -> D
A, B = NodeType(name="A", base=Cpd), NodeType(name="B", base=Cpd)
C, D = NodeType(name="C", base=Cpd), NodeType(name="D", base=Cpd)
R1, R2 = NodeType(name="R1", base=Rct), NodeType(name="R2", base=Rct)

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
spec = CDGSpec(cdg_types, production_rules, val_rules)
validator = ArkValidator(solver=SMTSolver())
validator.validate(cdg=graph, cdg_spec=spec)
compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[], import_lib={})

time_range = [0, 15]
time_points = np.linspace(*time_range, 1000)
mapping = compiler.var_mapping
init_states = compiler.map_init_state({a: 6, b: 2, c: 0, d: 0})
compiler.print_prog()
sol = compiler.prog(time_range, init_states=init_states, dense_output=True)
for node in [a, b, c]:
    idx = mapping[node]
    plt.plot(time_points, sol.sol(time_points)[idx].T, label=node.name)
plt.xlabel("time")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.savefig("examples/crn.png")
plt.show()
