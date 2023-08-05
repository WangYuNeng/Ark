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


# visualization scripts
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.graphviz_gen as graphvizlib


# Ark specification
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(min=0.5, max=2)

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
IdealE = EdgeType(name='E')
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
tln_lang.add_types(IdealV, IdealI, IdealE, InpV, InpI)
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
_v2i = ProdRule(IdealE, IdealV, IdealI, SRC, -VAR(DST)/SRC.c)
v2_i = ProdRule(IdealE, IdealV, IdealI, DST, VAR(SRC)/DST.l)
_i2v = ProdRule(IdealE, IdealI, IdealV, SRC, -VAR(DST)/SRC.l)
i2_v = ProdRule(IdealE, IdealI, IdealV, DST, VAR(SRC)/DST.c)
vself = ProdRule(IdealE, IdealV, IdealV, SELF, -VAR(SRC)*SRC.g/SRC.c)
iself = ProdRule(IdealE, IdealI, IdealI, SELF, -VAR(SRC)*SRC.r/SRC.l)
inpv2_v = ProdRule(IdealE, InpV, IdealV, DST, (SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
inpv2_i = ProdRule(IdealE, InpV, IdealI, DST, (SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
inpi2_v = ProdRule(IdealE, InpI, IdealV, DST, (SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
inpi2_i = ProdRule(IdealE, InpI, IdealI, DST, (SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
prod_rules = [_v2i, v2_i, _i2v, i2_v, vself, iself, inpv2_v, inpv2_i, inpi2_v, inpi2_i]

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
# latexlib.validation_rules_to_latex(tln_lang)

hw_tln_lang = CDGLang("hwtln",inherits=tln_lang)
# Nonideal implementation with 10% random variation
MmV = NodeType(name='Vm', base=IdealV,
               attr_def=[AttrDefMismatch('c', attr_type=float, attr_range=lc_range, rstd=0.1)])
MmI = NodeType(name='Im', base=IdealI,
               attr_def=[AttrDefMismatch('l', attr_type=float, attr_range=lc_range, rstd=0.1)])
MmE = EdgeType(name='Em', base=IdealE,
               attr_def=[AttrDefMismatch('ws', attr_type=float, attr_range=w_range, rstd=0.1),
                         AttrDefMismatch('wt', attr_type=float, attr_range=w_range, rstd=0.1)])
hw_tln_lang.add_types(MmV, MmI, MmE)

_v2i_mm = ProdRule(MmE, IdealV, IdealI, SRC, -EDGE.ws*VAR(DST)/SRC.c)
v2_i_mm = ProdRule(MmE, IdealV, IdealI, DST, EDGE.wt*VAR(SRC)/DST.l)
_i2v_mm = ProdRule(MmE, IdealI, IdealV, SRC, -EDGE.ws*VAR(DST)/SRC.l)
i2_v_mm = ProdRule(MmE, IdealI, IdealV, DST, EDGE.wt*VAR(SRC)/DST.c)
inpv2_v_mm = ProdRule(MmE, InpV, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
inpv2_i_mm = ProdRule(MmE, InpV, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
inpi2_v_mm = ProdRule(MmE, InpI, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
inpi2_i_mm = ProdRule(MmE, InpI, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
hw_prod_rules = [_v2i_mm, v2_i_mm, _i2v_mm, i2_v_mm, inpv2_v_mm, inpv2_i_mm, inpi2_v_mm, inpi2_i_mm]
hw_tln_lang.add_production_rules(*hw_prod_rules)
latexlib.production_rules_to_latex(hw_tln_lang)
latexlib.type_spec_to_latex(hw_tln_lang)

prod_rules += hw_prod_rules

cdg_types = [IdealV, IdealI, IdealE, InpV, InpI, MmV, MmI, MmE]
help_fn = [pulse]
spec = CDGSpec(cdg_types, prod_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())


def create_tline(v_nt: NodeType, i_nt: NodeType,
                 e_nt: EdgeType, line_len: int=10) \
                    -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    graph = CDG()
    if e_nt == IdealE:
        kwargs = {}
    elif e_nt == MmE:
        kwargs = {'ws': 1.0, 'wt': 1.0}
    current_in = InpI(fn=pulse, g=0.0)
    v_nodes = [v_nt(c=1e-9, g=0.0) for _ in range(line_len)] + [v_nt(c=1e-9, g=1.0)]
    i_nodes = [i_nt(l=1e-9, r=0.0) for _ in range(line_len)]
    for i in range(line_len):
        graph.connect(e_nt(**kwargs), v_nodes[i], i_nodes[i])
        graph.connect(e_nt(**kwargs), i_nodes[i], v_nodes[i + 1])
        graph.connect(IdealE(), v_nodes[i], v_nodes[i])
        graph.connect(IdealE(), i_nodes[i], i_nodes[i])
    graph.connect(e_nt(**kwargs), current_in, v_nodes[0])
    graph.connect(e_nt(**kwargs), v_nodes[-1], v_nodes[-1])
    return graph, v_nodes, i_nodes

if __name__ == '__main__':

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

        failed, _ = validator.validate(cdg=graph, cdg_spec=spec)
        if failed:
            raise ValueError("Validation failed!!")
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
