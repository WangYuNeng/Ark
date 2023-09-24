from types import FunctionType
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.range import Range
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG, CDGNode
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from ark.specification.validation_rule import ValRule, ValPattern
from ark.reduction import SUM
import sys

# visualization scripts
import ark.visualize.latex_gen as latexlib
import ark.visualize.graphviz_gen as graphvizlib


def subscr(name,idx):
    return "<%s<sub>%s</sub>>" % (name,idx)
    

# Ideal implementation
TargType = NodeType(name=subscr("NT","i"), order=1,
                    reduction=SUM,
                    attr_def=[])

n_types_rev = [subscr("NT","1"),subscr("NT","j"),subscr("NT","m")] 
NodeTypesRev = {}
EdgeTypesRev = {}
for name in n_types_rev:
    NodeTypesRev[name] = NodeType(name=name, order=1,
                    reduction=SUM,
                    attr_def=[])
                    
e_types_rev = [subscr("ET","1"),subscr("ET","j"),subscr("ET","m")] 
for name  in e_types_rev:
    EdgeTypesRev[name] = EdgeType(name=name,
                  attr_def=[])


n_types = [subscr("NT","m+1"),subscr("NT","k"),subscr("NT","n")] 
NodeTypes = {}
EdgeTypes = {}
for name in n_types:
    NodeTypes[name] = NodeType(name=name, order=1,
                    reduction=SUM,
                    attr_def=[])
                    
e_types = [subscr("ET","m+1"),subscr("ET","k"),subscr("ET","n")] 
for name  in e_types:
    EdgeTypes[name] = EdgeType(name=name,
                  attr_def=[])

cdg_lang = CDGSpec("tln")
cdg_lang.add_cdg_types(TargType)
cdg_lang.add_cdg_types(list(EdgeTypes.values()))
cdg_lang.add_cdg_types(list(EdgeTypesRev.values()))
cdg_lang.add_cdg_types(list(NodeTypes.values()))
cdg_lang.add_cdg_types(list(NodeTypesRev.values()))

graph = CDG()
targ = TargType() 
targ.name = targ.cdg_type.name
for i in range(len(e_types)):
    etyp = EdgeTypes[e_types[i]]()
    etyp.name = etyp.cdg_type.name
    ntyp = NodeTypes[n_types[i]]()
    ntyp.name = ntyp.cdg_type.name
    graph.connect(etyp, ntyp, targ)

e_types_rev.reverse()
for i in range(len(e_types_rev)):
    etyp = EdgeTypesRev[e_types_rev[i]]()
    etyp.name = etyp.cdg_type.name
    ntyp = NodeTypesRev[n_types_rev[i]]()
    ntyp.name = ntyp.cdg_type.name
    graph.connect(etyp, targ, ntyp)



def process(graph):
    style = {"style":"dotted", "penwidth":"5pt", "arrowhead":"none"}
    #graph.graph.edge(n_types[0],n_types[1],**style)
    #graph.graph.edge(n_types[1],n_types[2],**style)
    #graph.graph.edge(n_types_rev[0],n_types_rev[1],**style)
    #graph.graph.edge(n_types_rev[1],n_types_rev[2],**style)

    force_layout = False 
    if force_layout:
        graph.graph.graph_attr["layout"] = "neato"
        graph.graph.graph_attr["sep"] = "+7"
        graph.graph.graph_attr["overlap"] = "false"
        graph.graph.graph_attr["splines"] = "true"

    graph.graph.graph_attr["ratio"] = "1.0"


name = "cdg-example"
graphvizlib.cdg_to_graphviz("cdg",name,cdg_lang,graph,inherited=False,show_edge_labels=True,post_layout_hook=process)