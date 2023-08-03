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
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.graphviz_gen as graphvizlib


# Ark specification
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(exact=1.0)

tln_lang = CDGLang("tln")

# Ideal implementation
IdealV = NodeType(name='V', order=1,
                  reduction=SUM,
                  attr_def=[AttrDef('c', attr_type=float, attr_range=lc_range),
                         AttrDef('g', attr_type=float, attr_range=gr_range)
                        ])
IdealI = NodeType(name='I', order=1,
                  reduction=SUM,
                  attr_def=[AttrDef('l', attr_type=float, attr_range=lc_range),
                         AttrDef('r', attr_type=float, attr_range=gr_range)
                        ])
Meas = NodeType(name='Meas', order=0,
                  reduction=SUM,
                  attr_def=[])

IdealE = EdgeType(name='IdealE',
                  attr_def=[AttrDef('ws', attr_type=float,attr_range=w_range),
                         AttrDef('wt', attr_type=float,attr_range=w_range)
                        ])
InpV = NodeType(name='InpV',
                order=0,
                attr_def=[AttrDef('fn', attr_type=FunctionType),
                       AttrDef('r', attr_type=float, attr_range=gr_range)
                       ])
InpI = NodeType(name='InpI',
                order=0,
                attr_def=[AttrDef('fn', attr_type=FunctionType),
                          AttrDef('g', attr_type=float, attr_range=gr_range)
                          ])
tln_lang.add_types(IdealV, IdealI, IdealE, InpV, InpI,Meas)
latexlib.type_spec_to_latex(tln_lang)

# Example input function
def pulse(t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1):
    """Trapezoidal pulse function"""
    t = (t - delay) % period
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0


# Production rules
_v2i = ProdRule(IdealE, IdealV, IdealI, SRC, -EDGE.ws*VAR(DST)/SRC.c)
v2_i = ProdRule(IdealE, IdealV, IdealI, DST, EDGE.wt*VAR(SRC)/DST.l)
_i2v = ProdRule(IdealE, IdealI, IdealV, SRC, -EDGE.ws*VAR(DST)/SRC.l)
i2_v = ProdRule(IdealE, IdealI, IdealV, DST, EDGE.wt*VAR(SRC)/DST.c)
vself = ProdRule(IdealE, IdealV, IdealV, SELF, -VAR(SRC)*SRC.g/SRC.c)
iself = ProdRule(IdealE, IdealI, IdealI, SELF, -VAR(SRC)*SRC.r/SRC.l)
inpv2_v = ProdRule(IdealE, InpV, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
inpv2_i = ProdRule(IdealE, InpV, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
inpi2_v = ProdRule(IdealE, InpI, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
inpi2_i = ProdRule(IdealE, InpI, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
meas_m = ProdRule(IdealE, IdealV, Meas, DST, VAR(SRC))
meas_v = ProdRule(IdealE, IdealV, Meas, SRC, EDGE.wt*0.0)
prod_rules = [_v2i, v2_i, _i2v, i2_v, vself, iself, inpv2_v, inpv2_i, inpi2_v, inpi2_i, meas_m, meas_v]

tln_lang.add_production_rules(*prod_rules)
latexlib.production_rules_to_latex(tln_lang)

# Validation rules
v_val = ValRule(IdealV, [ValPattern(SRC, IdealE, IdealI, Range(min=0)),
                     ValPattern(DST, IdealE, IdealI, Range(min=0)),
                     ValPattern(DST, IdealE, InpV, Range(min=0)),
                     ValPattern(DST, IdealE, InpI, Range(min=0)),
                     ValPattern(SELF, IdealE, IdealV, Range(exact=1))])
i_val = ValRule(IdealI, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                     ValPattern(DST, IdealE, [IdealV, InpV, InpI], Range(min=0, max=1)),
                     ValPattern(SELF, IdealE, IdealI, Range(exact=1))])
inpv_val = ValRule(InpV, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                          ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1))])
inpi_val = ValRule(InpI, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                          ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1))])
val_rules = [v_val, i_val, inpv_val, inpi_val]
tln_lang.add_validation_rules(*val_rules)
latexlib.validation_rules_to_latex(tln_lang)

hw_tln_lang = CDGLang("hwtln",inherits=tln_lang)
# Nonideal implementation with 10% random variation
MmV = NodeType(name='MmV', base=IdealV,
               attr_def=[AttrDefMismatch('c', attr_type=float, attr_range=lc_range, rstd=0.1)])
MmI = NodeType(name='MmI', base=IdealI,
               attr_def=[AttrDefMismatch('l', attr_type=float, attr_range=lc_range, rstd=0.1)])
MmE = EdgeType(name='MmE', base=IdealE,
               attr_def=[AttrDefMismatch('ws', attr_type=float, attr_range=w_range, rstd=0.1)])
hw_tln_lang.add_types(MmV, MmI, MmE)
latexlib.type_spec_to_latex(hw_tln_lang)

cdg_types = [IdealV, IdealI, IdealE, InpV, InpI, MmV, MmI, MmE]
help_fn = [pulse]
spec = CDGSpec(cdg_types, prod_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())

def build_line(graph,e_nt, v_nt,i_nt,length,start_i=False):
    tc = 1e-9
    if start_i:
        v_nodes = [v_nt(c=tc, g=0.0) for _ in range(length)] +  [v_nt(c=tc, g=1.0)]
        i_nodes = [i_nt(l=tc, r=0.0) for _ in range(length+1)] 
        for i in range(length):
            graph.connect(e_nt(ws=1.0, wt=1.0), i_nodes[i], v_nodes[i])
            graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[i], i_nodes[i+1])
            graph.connect(IdealE(ws=1.0, wt=1.0), v_nodes[i], v_nodes[i])
            graph.connect(IdealE(ws=1.0, wt=1.0), i_nodes[i], i_nodes[i])
            
        graph.connect(e_nt(ws=1.0, wt=1.0), i_nodes[length], v_nodes[length])
        graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], v_nodes[-1])


    else:
        v_nodes = [v_nt(c=tc, g=0.0) for _ in range(length)] + [v_nt(c=tc, g=1.0)]
        i_nodes = [i_nt(l=tc, r=0.0) for _ in range(length)]
        for i in range(length):
            graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[i], i_nodes[i])
            graph.connect(e_nt(ws=1.0, wt=1.0), i_nodes[i], v_nodes[i + 1])
            graph.connect(IdealE(ws=1.0, wt=1.0), v_nodes[i], v_nodes[i])
            graph.connect(IdealE(ws=1.0, wt=1.0), i_nodes[i], i_nodes[i])

        graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], v_nodes[-1])

    return v_nodes, i_nodes


def create_tline_branch(v_nt: NodeType, i_nt: NodeType, e_nt: EdgeType,  line_len: int=5, \
                branch_len: int=2, branches_per_node: int=2, branch_stride: int=1):
    graph = CDG()
    current_in = InpI(fn=pulse, g=0.0)
    meas = Meas()
    v_nodes = []
    i_nodes = [] 
    v_nodes_line,i_nodes_line = build_line(graph,e_nt,v_nt, i_nt, line_len)
    v_nodes += v_nodes_line
    i_nodes += i_nodes_line

    assert(line_len % branch_stride == 0)
    total_branches =  branches_per_node*int(line_len/branch_stride)
    branches = {}
    for i in range(total_branches):
        v_nodes_branches,i_nodes_branches = build_line(graph,e_nt,v_nt,i_nt,branch_len,start_i=True)
        branches[i] = (v_nodes_branches,i_nodes_branches)
        v_nodes += v_nodes_branches
        i_nodes += i_nodes_branches


    idx = 0
    for i in range(0,line_len,branch_stride):
        targ_v = v_nodes_line[i]
        targ_i = i_nodes_line[i]
        for br in range(branches_per_node):
            br_v, br_i = branches[idx]
            graph.connect(e_nt(ws=1.0,wt=1.0), targ_v, br_i[0])
            idx += 1

    graph.connect(e_nt(ws=1.0, wt=1.0), current_in, v_nodes_line[0])
    graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes_line[-1], meas)
    v_nodes_line[0].name = "IN_V"
    v_nodes_line[-1].name = "OUT_V"


    return graph, v_nodes, i_nodes


def create_linear_tline(v_nt: NodeType, i_nt: NodeType,
                 e_nt: EdgeType, line_len=10) \
                    -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    graph = CDG()
    current_in = InpI(fn=pulse, g=0.0)
    meas = Meas()
    v_nodes = [v_nt(c=1e-9, g=0.0) for _ in range(line_len)] + [v_nt(c=1e-9, g=1.0)]
    i_nodes = [i_nt(l=1e-9, r=0.0) for _ in range(line_len)]
    for i in range(line_len):
        graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[i], i_nodes[i])
        graph.connect(e_nt(ws=1.0, wt=1.0), i_nodes[i], v_nodes[i + 1])
        graph.connect(IdealE(ws=1.0, wt=1.0), v_nodes[i], v_nodes[i])
        graph.connect(IdealE(ws=1.0, wt=1.0), i_nodes[i], i_nodes[i])
    graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], v_nodes[-1])

    graph.connect(e_nt(ws=1.0, wt=1.0), current_in, v_nodes[0])
    graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], meas)

    v_nodes[0].name = "IN_V"
    v_nodes[-1].name = "OUT_V"


    return graph, v_nodes, i_nodes

def nominal_simulation(cdg,time_range,name):
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
    plt.plot(time_points,trajs[in_traj_idx])
    plt.plot(time_points,trajs[out_traj_idx])

    filename = "gviz-output/tln-example/%s.pdf" % name
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':

    line_len = 12
    itl_linear, _, _ = create_linear_tline(IdealV, IdealI, MmE,line_len=10)
    graphvizlib.cdg_to_graphviz("tln-example","idl_tline_linear",hw_tln_lang,itl_linear,inherited=False, \
                horizontal=True,save_legend=True, show_node_labels=False)
    opts = {"nominal":True,"name":"idl_tline_linear"}

    branch_args = {"line_len":line_len, "branch_stride":4,"branches_per_node":1,"branch_len":7}
    itl_branch, _, _ = create_tline_branch(IdealV, IdealI, MmE,  **branch_args)
    graphvizlib.cdg_to_graphviz("tln-example","idl_tline_branch",hw_tln_lang,itl_branch,inherited=False, \
                horizontal=True,save_legend=False, show_node_labels=False)
    opts = {"nominal":True,"name":"idl_tline_branch"}

    TIME_RANGE = [0, 40e-9]
    for options, cdg_prog in [(opts,itl_linear), (opts,itl_branch)]:
        if options["nominal"]:
            nominal_simulation(cdg_prog,TIME_RANGE,options["name"])            

 

    sys.exit(0)
    mmn_tline, mmn_vs, mmn= create_tline(MmV, MmI, IdealE)
    name = "tline_%s" % handle
    mmn_tline, mme_vs ,mmn_is = create_tline(MmV, MmI, IdealE)

    LINE_LEN = 20
    N_RAND_SIM = 100
    TIME_RANGE = [0, 40e-9]
    fig, ax = plt.subplots(nrows=2)
    for color_idx, (vt, it, et, title,handle) in enumerate([(IdealV, IdealI, MmE, '10% Mismatched XXX',"mmG"),
                                                     (MmV, MmI, IdealE, '10 Mismatched LC',"mmLC"),
                                                     (IdealV, IdealI, IdealE, 'Ideal',"ideal")
                                                    ]):
        graph, v_nodes, i_nodes = create_tline(vt, it, et, LINE_LEN)

        name = "tline_%s" % handle
        graphvizlib.cdg_to_graphviz("tln",name,hw_tln_lang,graph,inherited=False)
        graphvizlib.cdg_to_graphviz("tln",name+"_inh",hw_tln_lang,graph,inherited=True)

        validator.validate(cdg=graph, cdg_spec=spec)
        compiler.compile(cdg=graph, cdg_spec=spec, help_fn=help_fn, import_lib={})
        mapping = compiler.var_mapping
        init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})

        for seed in range(N_RAND_SIM):
            sol = compiler.prog(TIME_RANGE, init_states=init_states, init_seed=seed, max_step=1e-10)
            time_points = sol.t
            trajs = sol.y
            for row_num, idx in enumerate([0, LINE_LEN]):
                traj_idx = mapping[v_nodes[idx]]
                if color_idx == 2:
                    alpha = 1.0
                else:
                    alpha = 0.5
                if seed == 0:
                    ax[row_num].plot(time_points * 1e9, trajs[traj_idx],
                                     color=f'C{color_idx}', alpha=alpha, label=f'{title}')
                else:
                    ax[row_num].plot(time_points * 1e9, trajs[traj_idx],
                                     color=f'C{color_idx}', alpha=alpha)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(reversed(handles), reversed(labels),loc='upper center', bbox_to_anchor=(0.5, 1.5),
          fancybox=True, shadow=True, ncol=3)

    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Amplitude (V)')
    ax[0].set_title('Source waveform')
    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Amplitude (V)')
    ax[1].set_title('Terminal waveform')
    plt.tight_layout()
    plt.savefig('examples/tln.png')
    plt.show()
