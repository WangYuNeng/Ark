"""
Example: Coupled Oscillator Network
https://www.nature.com/articles/s41598-019-49699-5
"""

import numpy as np
from types import FunctionType
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from ark.globals import Range, Attr
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.generation_rule import GenRule, SRC, DST, SELF, EDGE, VAR, TIME
from ark.reduction import SUM, PROD

Osc = NodeType(name='Osc', order=1, attr_defs={'lock_fn': Attr(attr_type=FunctionType),
                                               'osc_fn': Attr(attr_type=FunctionType)})
Coupling = EdgeType(name='Coupling', attr_defs={'k': Attr(attr_type=float)})

def locking_fn(t, tau=np.pi * 5):
    return np.exp(-t / tau)

def sin_fn(x):
    return np.sin(x)

rule1 = GenRule(Coupling, Osc, Osc, SRC, - EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST)))
rule2 = GenRule(Coupling, Osc, Osc, DST, - EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC)))
rule3 = GenRule(Coupling, Osc, Osc, SELF, - SELF.lock_fn(TIME) * SELF.osc_fn(VAR(SELF) * 2))

# MAXCUT Problem
a, b = Osc(lock_fn=locking_fn, osc_fn=sin_fn), Osc(lock_fn=locking_fn, osc_fn=sin_fn)
c, d = Osc(lock_fn=locking_fn, osc_fn=sin_fn), Osc(lock_fn=locking_fn, osc_fn=sin_fn)
e1 = Coupling(k=-1.0)
e2 = Coupling(k=-1.0)
e3 = Coupling(k=-1.0)
e_self = [Coupling(k=1.0) for _ in range(4)]

graph = CDG()
graph.connect(e1, a, b)
graph.connect(e2, a, c)
graph.connect(e3, a, d)
graph.connect(e_self[0], a, a)
graph.connect(e_self[1], b, b)
graph.connect(e_self[2], c, c)
graph.connect(e_self[3], d, d)

cdg_types = [Osc, Coupling]
generation_rules = [rule1, rule2, rule3]
spec = CDGSpec(cdg_types, generation_rules, None)
compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[locking_fn, sin_fn], import_lib={'np': np})

time_range = [0, 15]
time_points = np.linspace(*time_range, 1000)
mapping = compiler.var_mapping
states = [np.random.rand() * np.pi for _ in range(len(mapping))]
sol = solve_ivp(compiler.prog(), time_range, states, dense_output=True)
fig, ax = plt.subplots(nrows=len(states) + 1)
for node, idx in mapping.items():
    ax[-1].plot(time_points,
             node.attrs['osc_fn'](time_points + sol.sol(time_points)[idx].T),
             label=node.name)
    ax[idx].plot(time_points, sol.sol(time_points)[idx].T, label=node.name)
    ax[idx].legend()
# plt.xlabel('time')
# plt.ylabel('Value')
# plt.grid()
# plt.legend()
ax[-1].set_xlabel('time')
plt.tight_layout()
plt.show()
