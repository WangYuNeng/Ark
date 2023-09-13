'''
Example: TLN (Transmission Line nIdealEwork) CircuIdealI
Use LC ladders to emulate the telegrapher's equation
Provide specification for
- Ideal LC ladder
- LC mismatched ladder
- Gain mismatched ladder
'''
from types import FunctionType
import matplotlib.pyplot as plt
import matplotlib as mpl
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

import numpy as np

# visualization scripts
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.latex_gen_upd as latexlibnew
import ark.visualize.graphviz_gen as graphvizlib

from examples.tln import *

cdg_types = [IdealV, IdealI, IdealE, InpV, InpI, MmV, MmI, MmE]
help_fn = [pulse]
spec = CDGSpec(cdg_types, prod_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())

fontsize=25
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
    "font.family": "Helvetica",
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize
})

def build_line(graph,e_nt, v_nt,i_nt,length,term_g=1.0,start_i=False):
    tc = 1e-9
    if start_i:
        v_nodes = [v_nt(c=tc, g=0.0) for _ in range(length)] +  [v_nt(c=tc, g=term_g)]
        i_nodes = [i_nt(l=tc, r=0.0) for _ in range(length+1)] 
        for i in range(length):
            graph.connect(e_nt(), i_nodes[i], v_nodes[i])
            graph.connect(e_nt(), v_nodes[i], i_nodes[i+1])
            graph.connect(IdealE(), v_nodes[i], v_nodes[i])
            graph.connect(IdealE(), i_nodes[i], i_nodes[i])
            
        graph.connect(e_nt(), i_nodes[length], v_nodes[length])
        graph.connect(IdealE(), v_nodes[-1], v_nodes[-1])


    else:
        v_nodes = [v_nt(c=tc, g=0.0) for _ in range(length)] + [v_nt(c=tc, g=term_g)]
        i_nodes = [i_nt(l=tc, r=0.0) for _ in range(length)]
        for i in range(length):
            graph.connect(e_nt(), v_nodes[i], i_nodes[i])
            graph.connect(e_nt(), i_nodes[i], v_nodes[i + 1])
            graph.connect(IdealE(), v_nodes[i], v_nodes[i])
            graph.connect(IdealE(), i_nodes[i], i_nodes[i])

        graph.connect(IdealE(), v_nodes[-1], v_nodes[-1])
        v_nodes[-1].programmable = True

    return v_nodes, i_nodes


def create_tline_branch(v_nt: NodeType, i_nt: NodeType, e_nt: EdgeType,  e_nt_mm: EdgeType = None,
        mismatch_strategy = None, 
        line_len: int=5, branch_len: int=2, branches_per_node: int=2, 
        branch_offset: int=0, branch_stride: int=1):
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse, g=1.0)
    v_nodes = []
    i_nodes = []
    e_targ_nt = e_nt_mm if mismatch_strategy == "line-only"  and not e_nt_mm is None else e_nt 
    v_nodes_line,i_nodes_line = build_line(graph,e_targ_nt,v_nt, i_nt, line_len,term_g=1.0)
    v_nodes += v_nodes_line
    i_nodes += i_nodes_line

    assert(line_len % branch_stride == 0)
    total_branches =  branches_per_node*int(line_len/branch_stride)
    branches = {}
    for i in range(total_branches):
        e_targ_nt = e_nt_mm if mismatch_strategy == "branch-only" and not e_nt_mm is None else e_nt 
        v_nodes_branches,i_nodes_branches = build_line(graph,e_targ_nt,v_nt,i_nt,branch_len,start_i=True,term_g=0.0)
        branches[i] = (v_nodes_branches,i_nodes_branches)
        v_nodes += v_nodes_branches
        i_nodes += i_nodes_branches


    idx = 0
    for i in range(branch_offset,line_len,branch_stride):
        targ_v = v_nodes_line[i]
        targ_i = i_nodes_line[i]
        for br in range(branches_per_node):
            br_v, br_i = branches[idx]
            edge = e_nt()
            edge.switchable = True
            graph.connect(e_nt(), targ_v, br_i[0])
            idx += 1

    graph.connect(e_nt(), current_in, v_nodes_line[0])
    v_nodes_line[0].name = "IN_V"
    v_nodes_line[-1].name = "OUT_V"


    return graph, v_nodes, i_nodes


def create_malformed_tline(v_nt: NodeType, i_nt: NodeType,
                 e_nt: EdgeType, line_len=10) \
                    -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse, g=1.0)
    v_nodes = [v_nt(c=1e-9, g=0.0) for _ in range(line_len)] + [v_nt(c=1e-9, g=1.0)]
    i_nodes = [i_nt(l=1e-9, r=0.0) for _ in range(line_len)]
    for i in range(line_len):
        if i % 2 == 0:
            graph.connect(e_nt(), i_nodes[i], v_nodes[i])
            graph.connect(e_nt(), v_nodes[i], v_nodes[i + 1])
        else:
            graph.connect(e_nt(), v_nodes[i], i_nodes[i])
            graph.connect(e_nt(), i_nodes[i], v_nodes[i + 1])

        graph.connect(IdealE(), v_nodes[i], v_nodes[i])
        graph.connect(IdealE(), i_nodes[i], i_nodes[i])
    graph.connect(IdealE(), v_nodes[-1], v_nodes[-1])

    graph.connect(e_nt(), current_in, i_nodes[0])
    #graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], meas)

    v_nodes[0].name = "IN_V"
    v_nodes[-1].name = "OUT_V"


    return graph, v_nodes, i_nodes


def create_linear_tline(v_nt: NodeType, i_nt: NodeType,
                 e_nt: EdgeType, line_len=10) \
                    -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse, g=1.0)
    v_nodes = [v_nt(c=1e-9, g=0.0) for _ in range(line_len)] + [v_nt(c=1e-9, g=1.0)]
    i_nodes = [i_nt(l=1e-9, r=0.0) for _ in range(line_len)]
    for i in range(line_len):
        graph.connect(e_nt(), v_nodes[i], i_nodes[i])
        graph.connect(e_nt(), i_nodes[i], v_nodes[i + 1])
        graph.connect(IdealE(), v_nodes[i], v_nodes[i])
        graph.connect(IdealE(), i_nodes[i], i_nodes[i])
    graph.connect(IdealE(), v_nodes[-1], v_nodes[-1])

    graph.connect(e_nt(), current_in, v_nodes[0])
    #graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], meas)

    v_nodes[0].name = "IN_V"
    v_nodes[-1].name = "OUT_V"


    return graph, v_nodes, i_nodes

def nominal_simulation(cdg,time_range,name,post_process_hook=None):
    validator.validate(cdg=cdg, cdg_spec=spec)
    compiler.compile(cdg=cdg, cdg_spec=spec, help_fn=help_fn, import_lib={})
    mapping = compiler.var_mapping
    init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})
    sol = compiler.prog(time_range, init_states=init_states, init_seed=123, max_step=1e-10)
    time_points = sol.t
    trajs = sol.y
    in_node = list(filter(lambda n: n.name == "IN_V", cdg.nodes))[0]
    out_node = list(filter(lambda n: n.name == "OUT_V", cdg.nodes))[0]
    print(mapping)
    in_traj_idx = mapping[in_node]
    out_traj_idx = mapping[out_node]

    fig, ax = plt.subplots(1, 1, sharex=True)

    linecolor = "black"
    linewidth = 2.0
    
    plt.plot(time_points,trajs[out_traj_idx], color=linecolor, linewidth=linewidth)
    ax = plt.gca()
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_xticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
             xy=(1, -0.05), xycoords='axes fraction', fontsize=fontsize-5)
    if not post_process_hook is None:
        post_process_hook(fig,ax)

    filename = "gviz-output/tln-example/%s-plot.pdf" % name
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def mismatch_simulation(cdg,time_range,name,post_process_hook=None):
    N_RAND_SIM = 100

    validator.validate(cdg=cdg, cdg_spec=spec)
    compiler.compile(cdg=cdg, cdg_spec=spec, help_fn=help_fn, import_lib={})
    mapping = compiler.var_mapping

    fig, ax = plt.subplots(1, 1, sharex=True)


    init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})
    in_node = list(filter(lambda n: n.name == "IN_V", cdg.nodes))[0]
    out_node = list(filter(lambda n: n.name == "OUT_V", cdg.nodes))[0]
    in_traj_idx = mapping[in_node]
    out_traj_idx = mapping[out_node]

    alpha = 0.5
    linecolor = "black"
    linewidth = 2.0
    for seed in range(N_RAND_SIM):
        sol = compiler.prog(TIME_RANGE, init_states=init_states, init_seed=seed, max_step=1e-10)
        time_points = sol.t
        trajs = sol.y
        if seed == 0:
            ax.plot(time_points, trajs[out_traj_idx], alpha=alpha, color=linecolor, linewidth=linewidth)
        else:
            ax.plot(time_points , trajs[out_traj_idx], alpha=alpha, color=linecolor, linewidth=linewidth)

    ax = plt.gca()
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_xticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
             xy=(1, -0.05), xycoords='axes fraction', fontsize=fontsize-5)
    
    
    if not post_process_hook is None:
        post_process_hook(fig,ax)

    filename = "gviz-output/tln-example/%s-plot.pdf" % name
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def graph_process(graph):
    style = {"style":"dotted", "penwidth":"5pt", "arrowhead":"none"}
    
    graph.graph.graph_attr["layout"] = "neato"
    graph.graph.graph_attr["sep"] = "+7"
    graph.graph.graph_attr["overlap"] = "false"
    graph.graph.graph_attr["splines"] = "true"

def plot_process(fix,ax):
    pass 


def highlight_refl(fig,ax):
    WINDOW_SIZE = 40e-9
    xmin, xmax, ymin, ymax = plt.axis()
    wl = WINDOW_SIZE
    wh = 2*(WINDOW_SIZE)
    section = np.arange(wl,wh, (wh-wl)*0.01)
    shadecolor = "yellow"
    shadealpha = 0.2
    x = np.arange(xmin, xmax, (xmax-xmin)*0.01)
    ax.axvline(wl, color=shadecolor, lw=2, alpha=shadealpha)
    ax.axvline(wh, color=shadecolor, lw=2, alpha=shadealpha)
    ax.fill_between(section, ymin, ymax,  facecolor=shadecolor, alpha=shadealpha)
    ax.fill_between(section, ymin, ymax,  facecolor=shadecolor, alpha=shadealpha)

if __name__ == '__main__':
    fnargs = {"br":latexlibnew.SwitchArg("E_6", "br==1")}
    branch_args = {"line_len":2, "branch_stride":2,"branches_per_node":1,"branch_len":0,"branch_offset":0}
    itl_small, _, _ = create_tline_branch(IdealV, IdealI, lambda: IdealE(),  **branch_args)
    latexlibnew.gen_function("br-func",tln_lang,itl_small,fnargs,first_k=3)

    graphvizlib.cdg_to_graphviz("tln-example","idl-small",hw_tln_lang,itl_small,inherited=False, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    lin_opts = {"nominal":True,"name":"idl-tline-small", "post_process_hook":plot_process}

    itl_malform, _, _ = create_malformed_tline(IdealV, IdealI, lambda: IdealE(),line_len=2)
    graphvizlib.cdg_to_graphviz("tln-example","idl-malf",hw_tln_lang,itl_malform,inherited=False, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    lin_opts = {"nominal":True,"name":"idl-tline-small-malf", "post_process_hook":plot_process}



    LINE_LEN, BRANCH_LEN = 10, 16
    itl_linear, _, _ = create_linear_tline(IdealV, IdealI, lambda: IdealE(),line_len=LINE_LEN)
    graphvizlib.cdg_to_graphviz("tln-example","idl-tline-linear",hw_tln_lang,itl_linear,inherited=False, \
                horizontal=True,save_legend=True, show_node_labels=False, post_layout_hook=graph_process)
    lin_opts = {"nominal":True,"name":"idl-tline-linear", "post_process_hook":plot_process}

    branch_args = {"line_len":LINE_LEN, "branch_stride":LINE_LEN,"branches_per_node":1,"branch_len":BRANCH_LEN,"branch_offset":1}
    itl_branch, _, _ = create_tline_branch(IdealV, IdealI, lambda: IdealE(),  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","idl-tline-branch",hw_tln_lang,itl_branch,inherited=False, \
                horizontal=True,save_legend=False, show_node_labels=False, post_layout_hook=graph_process)
    br_opts = {"nominal":True,"name":"idl-tline-branch", "post_process_hook":highlight_refl}

    node_mm_branch, _, _ = create_tline_branch(MmV, MmI, lambda: IdealE(),  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","mmnode_tline_branch",hw_tln_lang,node_mm_branch,inherited=True, \
                horizontal=True,save_legend=False, show_node_labels=False)
    nodemm_br_opts = {"nominal":False,"name":"mmnode-tline-branch", "post_process_hook":plot_process}

    edge_mm_branch, _, _ = create_tline_branch(IdealV, IdealI, lambda: MmE(ws=1.0,wt=1.0),  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","mmedge-tline-branch",hw_tln_lang,edge_mm_branch,inherited=True, \
                horizontal=True,save_legend=False, show_node_labels=False, post_layout_hook=graph_process)
    edgemm_br_opts = {"nominal":False,"name":"mmedge-tline-branch", "post_process_hook":plot_process}

    
    edge_mmbranches_branch, _, _ = create_tline_branch(IdealV, IdealI, lambda: IdealE(), e_nt_mm=lambda: MmE(ws=1.0,wt=1.0), 
                mismatch_strategy="branch-only",  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","mmeBranches_tline_branch",hw_tln_lang,edge_mmbranches_branch,inherited=True, \
                horizontal=True,save_legend=False, show_node_labels=False, post_layout_hook=graph_process)
    emmbranch_opts = {"nominal":False,"name":"mmeBranches-tline-branch", "post_process_hook":plot_process}

 
    edge_mmline_branch, _, _ = create_tline_branch(IdealV, IdealI, lambda: IdealE(), \
                    e_nt_mm=lambda: MmE(ws=1.0,wt=1.0), mismatch_strategy="line-only",  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","mmeLine_tline_branch",hw_tln_lang,edge_mmline_branch,inherited=True, \
                horizontal=True,save_legend=False, show_node_labels=False, post_layout_hook=graph_process)
    emmline_opts = {"nominal":False,"name":"mmeLine-tline-branch", "post_process_hook":plot_process}


    itl_linear_short, _, _ = create_linear_tline(IdealV, IdealI, lambda: IdealE(),line_len=1)
    graphvizlib.cdg_to_graphviz("tln-example","idl-tline-linear-short",hw_tln_lang,itl_linear_short,inherited=False, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    lin_short_opts = {"nominal":True,"name":"idl-tline-linear-short", "post_process_hook":plot_process}

    emm_linear_short, _, _ = create_linear_tline(IdealV, IdealI, lambda: MmE(ws=1.0,wt=1.0),line_len=1)
    graphvizlib.cdg_to_graphviz("tln-example","emm-tline-linear-short",hw_tln_lang,emm_linear_short,inherited=True, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    emm_short_opts = {"nominal":True,"name":"emm-tline-linear-short", "post_process_hook":plot_process}

    nmm_linear_short, _, _ = create_linear_tline(MmV, MmI, lambda: IdealE(),line_len=1)
    graphvizlib.cdg_to_graphviz("tln-example","nmm-tline-linear-short",hw_tln_lang,nmm_linear_short,inherited=True, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    nmm_short_opts = {"nominal":True,"name":"nmm-tline-linear-short", "post_process_hook":plot_process}

    emm_linear, _, _ = create_linear_tline(IdealV, IdealI, lambda: MmE(ws=1.0,wt=1.0),line_len=LINE_LEN)
    graphvizlib.cdg_to_graphviz("tln-example","emm-tline-linear",hw_tln_lang,emm_linear, inherited=True, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    emm_opts = {"nominal":False,"name":"emm-tline-linear", "post_process_hook":plot_process}

    nmm_linear, _, _ = create_linear_tline(MmV, MmI, lambda: IdealE(),line_len=LINE_LEN)
    graphvizlib.cdg_to_graphviz("tln-example","nmm-tline-linear",hw_tln_lang,nmm_linear,inherited=True, \
                horizontal=True,save_legend=True, show_node_labels=True, post_layout_hook=None)
    nmm_opts = {"nominal":False,"name":"nmm-tline-linear", "post_process_hook":plot_process}

    WINDOWS = 2
    TIME_RANGE = [0, 40e-9*WINDOWS]
    for options, cdg_prog in [(lin_opts,itl_linear), (br_opts,itl_branch), \
                                (nodemm_br_opts,node_mm_branch), (edgemm_br_opts,edge_mm_branch), 
                                (emmbranch_opts, edge_mmbranches_branch), (emmline_opts, edge_mmline_branch),
                                (lin_short_opts,itl_linear_short), (emm_short_opts, emm_linear_short), (nmm_short_opts, nmm_linear_short),
                                (emm_opts, emm_linear), (nmm_opts, nmm_linear)]:
        if options["nominal"]:
            nominal_simulation(cdg_prog,TIME_RANGE,options["name"],post_process_hook=options["post_process_hook"])            
        else:
            mismatch_simulation(cdg_prog,TIME_RANGE,options["name"],post_process_hook=options["post_process_hook"])            

 
