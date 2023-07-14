'''
Example: Chemical Reaction Network
https://link.springer.com/article/10.1007/s11047-019-09775-1
'''
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.attribute import Range, Attr
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.generation_rule import GenRule, SRC, DST, SELF, EDGE, VAR, TIME
from ark.reduction import SUM, PROD

Cpd = NodeType(name='Cpd', order=1)
Rct = NodeType(name='Rct', order=0, reduction=PROD,
               attr_defs={'k': Attr(attr_type=float,
                                    attr_range=Range(min=0.0, max=1.0))
                        }
               )
Rct_et = EdgeType(name='Rct_et', attr_defs={'nc': Attr(attr_type=int),
                                            'coeff': Attr(attr_type=int, attr_range=Range(min=1))
                                            }
                  )

rule1 = GenRule(Rct_et, Cpd, Rct, SRC, DST.k * EDGE.nc * VAR(DST))
rule2 = GenRule(Rct_et, Rct, Cpd, DST, SRC.k * EDGE.nc * VAR(SRC))
rule3 = GenRule(Rct_et, Cpd, Rct, DST, VAR(SRC) ** EDGE.coeff)

# Example: Multiplication C = A * B
# Compound A, B, C, D
# Reaction 1: A + B -> A + B + C
# Reaction 2: C -> D
A, B = NodeType(name='A', base=Cpd), NodeType(name='B', base=Cpd)
C, D = NodeType(name='C', base=Cpd), NodeType(name='D', base=Cpd)
R1, R2 = NodeType(name='R1', base=Rct), NodeType(name='R2', base=Rct)


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
generation_rules = [rule1, rule2, rule3]
spec = CDGSpec(cdg_types, generation_rules, None)
compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[], import_lib={})

time_range = [0, 15]
time_points = np.linspace(*time_range, 1000)
mapping = compiler.var_mapping
states = [0 for _ in mapping]
states[mapping[a]] = 6
states[mapping[b]] = 2
states[mapping[c]] = 0
sol = solve_ivp(compiler.prog(), time_range, states, dense_output=True)
for node in [a, b, c]:
    idx = mapping[node]
    plt.plot(time_points, sol.sol(time_points)[idx].T, label=node.name)
plt.xlabel('time')
plt.ylabel('Value')
plt.grid()
plt.legend()
plt.show()
