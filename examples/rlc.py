'''
Example: RLC Circuit
'''
from types import FunctionType
import inspect
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
# NodeType
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(min=1, max=1)
V_base = NodeType(name='V_base', order=1,
                  reduction=SUM,
                  attr_defs={'c': Attr(attr_type=float,attr_range=lc_range),
                             'g': Attr(attr_type=float, attr_range=gr_range)
                             })
I_base = NodeType(name='I_base', order=1,
                  reduction=PROD,
                  attr_defs={'l': Attr(attr_type=float, attr_range=lc_range),
                             'r': Attr(attr_type=float, attr_range=gr_range)
                            })
E_base = EdgeType(name='E_base',
                  attr_defs={'ws': Attr(attr_type=float,attr_range=w_range),
                             'wt': Attr(attr_type=float,attr_range=w_range)
                             })
InpV = NodeType(name='InpV',
                attr_defs={'fn': Attr(attr_type=FunctionType),
                           'r': Attr(attr_type=float, attr_range=gr_range)
                           })

V_derive = NodeType(name='V_derive', base=V_base)
V_derive2 = NodeType(name='V_derive2', base=V_derive)
I_derive = NodeType(name='I_derive', base=I_base)

r0 = GenRule(E_base, I_base, V_base, DST, -EDGE.ws*VAR(SRC)/SRC.l)
r1 = GenRule(E_base, I_base, V_base, SRC, -EDGE.wt*VAR(DST)/SRC.l)
r2 = GenRule(E_base, I_base, V_derive2, SRC, -EDGE.wt*VAR(DST)/SRC.l)
r3 = GenRule(E_base, I_derive, V_derive, SRC, -EDGE.wt*VAR(DST)/SRC.l)
r4 = GenRule(E_base, V_derive, V_derive, SELF, VAR(DST))
r5 = GenRule(E_base, InpV, V_base, DST, SRC.fn(TIME))
print(r5)

def test_input(t):
    return 10 * t

v1 = V_base(c=1e-9, g=0.0)
v2 = V_base(c=1e-9, g=0.0)
i1 = I_base(l=1e-9, r=0.0)
e1 = E_base(ws=1.0, wt=1.0)
e2 = E_base(ws=1.0, wt=1.0)
e3 = E_base(ws=1.0, wt=1.0)
e4 = E_base(ws=1.0, wt=1.0)
e5 = E_base(ws=1.0, wt=1.0)
inp = InpV(fn=test_input, r=0.0)
print(test_input.__name__)
sig = inspect.signature(test_input)
print(inspect.signature(test_input))

i2 = I_derive(l=1e-9, r=0.0)
v3 = V_derive(c=1e-9, g=0.0)
v4 = V_derive2(c=1e-9, g=0.0)

graph = CDG()
graph.connect(e1, i1, v1)
graph.connect(e2, i1, v2)
graph.connect(e3, i2, v1)
graph.connect(e4, v3, v3)
graph.connect(e5, inp, v3)

cdg_types = [V_base, I_base, E_base, InpV]
generation_rules = [r0, r1, r2, r3, r4, r5]
spec = CDGSpec(cdg_types, generation_rules, None)
print(spec.match_gen_rule(e1, i1, v1, SRC))
print(spec.match_gen_rule(e1, i1, v3, SRC))
print(spec.match_gen_rule(e1, i1, v4, SRC))
print(spec.match_gen_rule(e1, i2, v3, SRC))
# print(spec.match_gen_rule(e1, i2, v4, SRC))

compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[test_input], import_lib={})
