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
vs = graph.add_node(name='vs', cdg_type=S, attrs={'fn': 'sin(t)', 'r':'1'})
l1 = graph.add_node(name='l1', cdg_type=IN, attrs={'l': 1e-9})
c1 = graph.add_node(name='c1', cdg_type=VN, attrs={'c': 1e-9})
rl = graph.add_node(name='rl', cdg_type=R, attrs={'r': 1})
e1 = graph.add_edge(name='e1', cdg_type=E, attrs={'q_src': 1, 'q_dst': 1}, src=vs, dst=l1)
e2 = graph.add_edge(name='e2', cdg_type=E, attrs={'q_src': 1, 'q_dst': 1}, src=l1, dst=c1)
e3 = graph.add_edge(name='e3', cdg_type=E, attrs={'q_src': 1, 'q_dst': 1}, src=c1, dst=rl)

# validate
validator.validate(cdg=graph, cdg_spec=spec)

# compile
compiler.compile(cdg=graph, cdg_spec=spec)