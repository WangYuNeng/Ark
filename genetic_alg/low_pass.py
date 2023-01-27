# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:12:29 2023

@author: zousa
"""
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

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
from ark.specification.types import NodeType, StatefulNodeType, EdgeType
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
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=IN, gen_tgt=GenRule.DST, fn_exp='1/DST.l/(E.q_src*SRC.fn-SRC.r*DST)'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=IN, gen_tgt=GenRule.SRC, fn_exp='-DST/SRC.c*E.q_src'))
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='SRC/DST.c*E.q_dst'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=R, gen_tgt=GenRule.SRC, fn_exp='-SRC/SRC.c/DST.r'))
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='(E.q_src*SRC.fn-DST)/DST.c/SRC.r'))

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

# CDG
graph = CDG()
vs = graph.add_node(name='vs', cdg_type=S, attrs={'fn': 'mixed_sin(t)', 'r':'1'}) # Says Sin is not defined? How to put in your own signal?
# l1 = graph.add_node(name='l1', cdg_type=IN, attrs={'l': '1e-9'})
c1 = graph.add_node(name='c1', cdg_type=VN, attrs={'c': '1e-4'})
# rl = graph.add_node(name='rl', cdg_type=R, attrs={'r': '1'})
e1 = graph.add_edge(name='e1', cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=vs, dst=c1)
# e2 = graph.add_edge(name='e2', cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=l1, dst=c1)
# e3 = graph.add_edge(name='e3', cdg_type=E, attrs={'q_src': '1', 'q_dst': '1'}, src=c1, dst=rl)

# Helper Function
audiofile = 'genetic_alg/cut_sample.ogg'
data, samplerate = sf.read(audiofile)

def audio(t):
    samplerate = 32000
    period = 1/samplerate
    p1 = data[int(t*samplerate)]
    p2 = data[int(t*samplerate)+1]
    theta = (t/period - int(t/period))/period
    return (1-theta)*p1 + theta*p2
    
def pulse(t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1):
    t = (t - delay) % period 
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0

def mixed_sin(t):
    f1 = 1000
    f2 = 10000
    return np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
    
def dynamics(t, __VARIABLES):
    (c1,) = __VARIABLES
    (c1_c,) = (1e-04,)
    (vs_fn, vs_r) = (mixed_sin(t), 1)
    (e1_q_src, e1_q_dst) = (1, 1)
    ddt_c1 = (e1_q_src * vs_fn - c1) / c1_c / vs_r
    return [ddt_c1]

# validate
validator.validate(cdg=graph, cdg_spec=spec)

# compile
compiler.compile(cdg=graph, cdg_spec=spec, help_fn = [mixed_sin], import_lib={'sf': sf, 'data': data, 'np': np})

n_states = 1
time_range = [0, .001]
time_points = np.linspace(*time_range, 100000)
states = [0 for _ in range(n_states)]
sol = solve_ivp(compiler.prog(), time_range, states, dense_output=True)
plt.plot(time_points, [mixed_sin(t) for t in time_points], label = 'input')
plt.plot(time_points, np.sin(2*np.pi*1000*time_points), label = "1e4 Hz")
plt.plot(time_points, np.sin(2*np.pi*10000*time_points), label = "1e5 Hz")
plt.plot(time_points, sol.sol(time_points).T, label = "output")
plt.xlabel('time(s)')
plt.ylabel('Amplitude(V)')
plt.legend()
plt.savefig('tmp.png')
plt.clf()




