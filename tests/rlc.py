from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.globals import Direction
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG

compiler = ArkCompiler(rewrite=RewriteGen())
validator = ArkValidator(solver=SMTSolver())

# CDGType
from ark.specification.cdg_types import NodeType, StatefulNodeType, EdgeType
VN = StatefulNodeType(type_name='VN', attrs={'c': [0.1e-9, 10e-9]})
IN = StatefulNodeType(type_name='IN', attrs={'l': [0.1e-9, 10e-9]})
R = NodeType(type_name='R', attrs={'r': [0, 1e6]})
S = NodeType(type_name='S', attrs={'fn': ['func', -2, 2], 'r': [0, 1e6]})
E = EdgeType(type_name='E', attrs={'q_src': [0.5, 1.5], 'q_dst': [0.5, 1.5]})
cdg_types = [VN, IN, R, S, E]

# Generation Rules
from ark.specification.generation_rule import GenRule
gen_rules = []
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=VN, gen_tgt=GenRule.SRC, fn_exp='-E.q_src*DST/SRC.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=IN, gen_tgt=GenRule.DST, fn_exp='E.q_dst*SRC/DST.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=R, gen_tgt=GenRule.SRC, fn_exp='-DST.r*SRC/SRC.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=IN, gen_tgt=GenRule.DST, fn_exp='1/DST.l*(E.q_dst*SRC.fn-SRC.r*DST)'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=IN, gen_tgt=GenRule.SRC, fn_exp='-DST/SRC.c*E.q_src'))
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='SRC/DST.c*E.q_dst'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=R, gen_tgt=GenRule.SRC, fn_exp='-SRC/SRC.c/DST.r'))
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='(E.q_dst*SRC.fn-DST)/DST.c/SRC.r'))

# Validation Rules
from ark.specification.validation_rule import ValRule, DegreeConstraint, Connection
inf_conn = DegreeConstraint('*')
one_conn = DegreeConstraint('=1')
le_one_conn = DegreeConstraint('<=1')
VN_rule = ValRule(tgt_node_type=VN, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=inf_conn, node_types=[IN]),
    Connection(edge_type=E, direction=Direction.IN, degree=inf_conn, node_types=[IN, S]),
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[R])
])
IN_rule = ValRule(tgt_node_type=IN, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[VN]),
    Connection(edge_type=E, direction=Direction.IN, degree=le_one_conn, node_types=[VN, S]),
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[R]),
])
S_rule = ValRule(tgt_node_type=S, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=one_conn, node_types=[VN, IN])
])
R_rule = ValRule(tgt_node_type=R, connections=[
    Connection(edge_type=E, direction=Direction.IN, degree=one_conn, node_types=[VN, IN])
])
val_rules = [VN_rule, IN_rule, S_rule, R_rule]

# Specification
spec = CDGSpec(cdg_types=cdg_types, generation_rules=gen_rules, validation_rules=val_rules)

# Helper Function
def pulse(t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1):
    t = (t - delay) % period 
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0

# CDG
graph = CDG()
vs = graph.add_node(name='vs', cdg_type=S, attrs={'fn': 'pulse(t)', 'r':'1'})
n_ladder = 20
l = graph.add_node(name='l{}'.format(0), cdg_type=IN, attrs={'l': '1e-9'})
e = graph.add_edge(name='es_l0', cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=vs, dst=l)
for i in range(n_ladder - 1):
    c = graph.add_node(name='c{}'.format(i), cdg_type=VN, attrs={'c': '1e-9'})
    e1 = graph.add_edge(name='e_l{}_c{}'.format(i, i), cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=l, dst=c)
    l = graph.add_node(name='l{}'.format(i + 1), cdg_type=IN, attrs={'l': '1e-9'})
    e2 = graph.add_edge(name='e_c{}_l{}'.format(i, i+1), cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=c, dst=l)
c = graph.add_node(name='c{}'.format(n_ladder - 1), cdg_type=VN, attrs={'c': '1e-9'})
e = graph.add_edge(name='e_l{}_c{}'.format(n_ladder - 1, n_ladder - 1), cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=l, dst=c)
r = graph.add_node(name='r', cdg_type=R, attrs={'r': '1e3'})
e3 = graph.add_edge(name='e_c{}_r'.format(n_ladder - 1), cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=c, dst=r)

# validate
validator.validate(cdg=graph, cdg_spec=spec)

# compile
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[pulse], import_lib={})

n_states = n_ladder * 2
time_range = [0, 100e-9]
time_points = np.linspace(*time_range, 1000)
states = [0 for _ in range(n_states)]
sol = solve_ivp(compiler.prog(), time_range, states, dense_output=True, max_step=1e-10)
# print(sol.sol(time_points))
plt.plot(time_points, [pulse(t) for t in time_points])
for i in range(2):
    plt.plot(time_points, sol.sol(time_points)[i].T)
plt.xlabel('time(s)')
plt.ylabel('Amplitude(V)')
plt.savefig('tmp.png')
plt.clf()