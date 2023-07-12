'''
Example: RLC Circuit
'''
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from types import FunctionType
from ark.globals import Range, Attr
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.generation_rule import GenRule, SRC, DST, EDGE, VAR, SELF
# NodeType
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(min=1, max=1)
V_base = NodeType(name='V_base', order=1,
                  attr_defs={'c': Attr(attr_type=float,attr_range=lc_range),
                             'g': Attr(attr_type=float, attr_range=gr_range)
                             })
I_base = NodeType(name='I_base', order=1,
                  attr_defs={'l': Attr(attr_type=float, attr_range=lc_range),
                             'r': Attr(attr_type=float, attr_range=gr_range)
                            })
E_base = EdgeType(name='E_base',
                  attr_defs={'ws': Attr(attr_type=float,attr_range=w_range),
                             'wt': Attr(attr_type=float,attr_range=w_range)
                             })
InpV = EdgeType(name='InpV',
                attr_defs={'fn': Attr(attr_type=FunctionType),
                           'r': Attr(attr_type=float, attr_range=gr_range)
                           })

r0 = GenRule(E_base, I_base, V_base, DST, -EDGE.ws*VAR(SRC)/SRC.l)
r1 = GenRule(E_base, I_base, V_base, SRC, -EDGE.wt*VAR(DST)/SRC.l)

print(r0.fn_ast)
print(r0.identifier)

v1 = V_base(c=1e-9, g=0.0)
v2 = V_base(c=1e-9, g=0.0)
i1 = I_base(l=1e-9, r=0.0)
e1 = E_base(ws=1.0, wt=1.0)

graph = CDG()
graph.connect(e1, i1, v1)
print(v1.name, v2.name, i1.name, e1.name)
print(e1.src, e1.dst)
print(v1.degree)

cdg_types = [V_base, I_base, E_base, InpV]
generation_rules = [r0, r1]
spec = CDGSpec(cdg_types, generation_rules, None)
compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=graph, cdg_spec=spec)
