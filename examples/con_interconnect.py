"""
Example: Coupled Oscillator Network with Possible Interconnects Options
"""

from types import FunctionType
from argparse import ArgumentParser
from itertools import product
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.specification import CDGSpec
from ark.specification.range import Range
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.validation_rule import ValRule, ValPattern
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME

# visualization scripts
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.latex_gen_upd as latexlibnew
import ark.visualize.graphviz_gen as graphvizlib

N_GROUP = 4
con_lang = CDGLang("con")
hw_con_lang = CDGLang("intercon-con", inherits=con_lang)


Osc = NodeType(name='Osc', order=1, attr_def=[AttrDef('lock_fn', attr_type=FunctionType, nargs=1),
                                               AttrDef('osc_fn', attr_type=FunctionType, nargs=1)
                                               ])
Coupling = EdgeType(name='Cpl', attr_def=[AttrDef('k', attr_type=float)])

Osc_group = [NodeType(name=f'Osc_G{i}', base=Osc) for i in range(N_GROUP)]
Coupling_local = EdgeType(name='Cpl_l', base=Coupling)
Coupling_global = EdgeType(name='Cpl_g', base=Coupling)

con_lang.add_types(Osc, Coupling)
hw_con_lang.add_types(*Osc_group, Coupling_local, Coupling_global)
latexlib.type_spec_to_latex(hw_con_lang)

def locking_fn(x):
    """Injection locking function"""
    return 2 * 795.8e6 * np.sin(2 * np.pi * x)

def coupling_fn(x):
    """Coupling function"""
    return 2 * 795.8e6 * np.sin(np.pi * x)


r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, - EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST)))
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, - EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC)))
r_lock = ProdRule(Coupling, Osc, Osc, SELF, - SRC.lock_fn(TIME, VAR(SRC)))

val_rules = []
for i in range(N_GROUP):
    node_type = Osc_group[i]
    val_rules.append(ValRule(node_type, [ValPattern(SELF, Coupling_local, node_type, Range(exact=1)),
                                         ValPattern(SRC, Coupling_local, node_type, Range(min=0)),
                                         ValPattern(DST, Coupling_local, node_type, Range(min=0)),
                                         ValPattern(SRC, Coupling_global, Osc, Range(min=0)),
                                         ValPattern(DST, Coupling_global, Osc, Range(min=0))]
                            ))
hw_con_lang.add_validation_rules(*val_rules)
latexlib.validation_rules_to_latex(hw_con_lang)

latexlibnew.language_to_latex(hw_con_lang)

cdg_types = [Osc, Coupling] + Osc_group
production_rules = [r_cp_src, r_cp_dst, r_lock]
help_fn = [locking_fn, coupling_fn]
cdg_spec = CDGSpec(cdg_types, production_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())

N_NODES = 3

def validate_and_compile(graph, cdg_spec):
    failed, cex = validator.validate(graph, cdg_spec)
    if not failed:
        compiler.compile(graph, cdg_spec, import_lib={}, help_fn=help_fn)
        print('Validation and compilation successful')
    else:
        print('Validation failed')

def test_graph_0():
    """Normal case, local edge to connect within groups and
    global edge to connect across groups, should pass validation"""
    graph = CDG()
    grouped_nodes = [[Osc_group[i](lock_fn=locking_fn, osc_fn=coupling_fn)
                      for _ in range(N_NODES)] for i in range(N_GROUP)]
    for group, nodes in enumerate(grouped_nodes):
        for node_id, node in enumerate(nodes):
            graph.connect(Coupling_local(k=-1.0), node, nodes[(node_id + 1) % N_NODES])
            graph.connect(Coupling_local(k=1.0), node, node)
        graph.connect(Coupling_global(k=-1.0), nodes[0], grouped_nodes[(group + 1) % N_GROUP][0])

    validate_and_compile(graph, cdg_spec)

def test_graph_1():
    """Use local coupling for nodes across groups, should fail validation"""
    graph = CDG()
    grouped_nodes = [[Osc_group[i](lock_fn=locking_fn, osc_fn=coupling_fn)
                      for _ in range(N_NODES)] for i in range(N_GROUP)]
    for group, nodes in enumerate(grouped_nodes):
        for node_id, node in enumerate(nodes):
            graph.connect(Coupling_local(k=-1.0), node, nodes[(node_id + 1) % N_NODES])
            graph.connect(Coupling_local(k=1.0), node, node)
        graph.connect(Coupling_local(k=-1.0), nodes[0], grouped_nodes[(group + 1) % N_GROUP][0])

    validate_and_compile(graph, cdg_spec)

def main():
    test_graph_0()
    test_graph_1()

if __name__ == '__main__':
    main()
