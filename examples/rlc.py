'''
Example: RLC Circuit
'''
# from ark.compiler import ArkCompiler
# from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from types import FunctionType
from ark.globals import Range, Attr, MismatchAttr
# from ark.specification.specification import CDGSpec
# from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType, CDGType

# NodeType
V = NodeType(name='V', order=1, attr_defs={'c': Attr(attr_type=float,
                                    attr_range=Range(min=0.1e-9, max=10e-9)),
                                'g': Attr(attr_type=float, attr_range=Range(min=0))
                            })
'src.l * edg.g'



v1 = V(c=1e-9, g=0.0)
V_derived = NodeType(name='V_derived', inherit=V,
                     attr_defs={'d': Attr(attr_type=float, attr_range=Range(min=0))})
v2 = V_derived(c=1e-9, g=0.0, d=1.0)
v3 = V(c=1e-9, g=0.0)
# print(v2.c)
# I = NodeType(order=1, attrs={'l': Attr(float, ValidRange(min=0.1e-9, max=10e-9)),
#                                 'r': Attr(float, ValidRange(min=0))})
# Va = NodeType(parent_type=V, attrs={'area': Attr(int, 1)})
# Ia = NodeType(parent_type=I, attrs={'area': Attr(int, 1)})
# # TODO: how to specify the type of the function?
# InpV = NodeType(order=0, attrs={'fn':  Attr(type, None), 'r': Attr(float, ValidRange(min=0))})

# # EdgeType
# E = EdgeType(attrs={'ws': [float, 1], 'wt': [float, 1]})
# Ea = EdgeType(parent_type=E, attrs={'ws': [float, MismatchAttr()], 'wt': [float, 1]})

# assert isinstance(v_nodes[0], V)

