import numpy as np

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
R = NodeType(type_name='R', attrs={'r': [0.5, 1.5]})
S = NodeType(type_name='S', attrs={'fn': ['func', -2, 2], 'r': [0.5, 1.5]})
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

class LadderModel:

    _cdg_types = cdg_types
    _spec = spec
    _help_fn = [pulse]
    _param_ranges = {
        VN: VN.attrs,
        IN: IN.attrs,
        R: R.attrs,
        E: E.attrs,
        S: {'amplitude': [-5, 5], 'delay': [0, 50e-9], 'rise_time': [0, 50e-9], 'fall_time': [0, 50e-9], 
            'pulse_width': [0, 50e-9], 'period': [1, 2], 'r': [0.5, 1.5]} # arbitrary param range only for random testing
    }

    
    def get_param_range(self, cdg_type) -> dict:
        return self._param_ranges[cdg_type]

    @property
    def VN(self):
        return self._cdg_types[0]

    @property
    def IN(self):
        return self._cdg_types[1]

    @property
    def R(self):
        return self._cdg_types[2]

    @property
    def S(self):
        return self._cdg_types[3]
    
    @property
    def E(self):
        return self._cdg_types[4]

    @property
    def spec(self):
        return self._spec

    @property
    def help_fn(self):
        return self._help_fn

