'''
Example: RLC Circuit
'''
from types import FunctionType
import inspect
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.range import Range
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from ark.specification.validation_rule import ValRule
from ark.reduction import SUM
# NodeType
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(exact=1.0)
Vt = NodeType(name='Vt', order=1,
                  reduction=SUM,
                  attr_def=[AttrDef('c', attr_type=float,attr_range=lc_range),
                         AttrDef('g', attr_type=float, attr_range=gr_range)
                        ])
It = NodeType(name='It', order=1,
                  reduction=SUM,
                  attr_def=[AttrDef('l', attr_type=float, attr_range=lc_range),
                         AttrDef('r', attr_type=float, attr_range=gr_range)
                        ])
Et = EdgeType(name='Et',
                  attr_def=[AttrDef('ws', attr_type=float,attr_range=w_range),
                         AttrDef('wt', attr_type=float,attr_range=w_range)
                        ])
InpV = NodeType(name='InpV',
                attr_def=[AttrDef('fn', attr_type=FunctionType),
                       AttrDef('r', attr_type=float, attr_range=gr_range)
                       ])
InpI = NodeType(name='InpI',
                attr_def=[AttrDef('fn', attr_type=FunctionType),
                          AttrDef('g', attr_type=float, attr_range=gr_range)
                          ])
_v2i = ProdRule(Et, Vt, It, SRC, -EDGE.ws*VAR(DST)/SRC.c)
v2_i = ProdRule(Et, Vt, It, DST, EDGE.wt*VAR(SRC)/DST.l)
_i2v = ProdRule(Et, It, Vt, SRC, -EDGE.ws*VAR(DST)/SRC.l)
i2_v = ProdRule(Et, It, Vt, DST, EDGE.wt*VAR(SRC)/DST.c)
vself = ProdRule(Et, Vt, Vt, SELF, -VAR(SRC)*SRC.g/SRC.c)
iself = ProdRule(Et, It, It, SELF, -VAR(SRC)*SRC.r/SRC.l)
inpv2_v = ProdRule(Et, InpV, Vt, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
inpv2_i = ProdRule(Et, InpV, It, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
inpi2_v = ProdRule(Et, InpI, Vt, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
inpi2_i = ProdRule(Et, InpI, It, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
prod_rules = [_v2i, v2_i, _i2v, i2_v, vself, iself, inpv2_v, inpv2_i, inpi2_v, inpi2_i]



def test_input(t):
    return 10 * t

v1 = Vt(c=1e-9, g=0.0)
v2 = Vt(c=1e-9, g=0.0)
i1 = It(l=1e-9, r=0.0)
e1 = Et(ws=1.0, wt=1.0)
e2 = Et(ws=1.0, wt=1.0)
e3 = Et(ws=1.0, wt=1.0)
e4 = Et(ws=1.0, wt=1.0)
e5 = Et(ws=1.0, wt=1.0)
inp = InpV(fn=test_input, r=0.0)
print(test_input.__name__)
sig = inspect.signature(test_input)
print(inspect.signature(test_input))

i2 = It(l=1e-9, r=0.0)
v3 = Vt(c=1e-9, g=0.0)
v4 = Vt(c=1e-9, g=0.0)

graph = CDG()
graph.connect(e1, i1, v1)
graph.connect(e2, i1, v2)
graph.connect(e3, i2, v1)
graph.connect(e4, v3, v3)
graph.connect(e5, inp, v3)

cdg_types = [Vt, It, Et, InpV, InpI]
spec = CDGSpec(cdg_types, prod_rules, None)

compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec, help_fn=[test_input], import_lib={})
compiler.print_prog()
