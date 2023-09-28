import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ark.cdg.cdg import CDG, CDGNode
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen, SympyRewriteGen
from ark.solver import SMTSolver
from ark.specification.cdg_types import NodeType, EdgeType
from ark.validator import ArkValidator
from spec import tln_spec, mm_tln_spec, pulse_sympy

# visualization scripts
import ark.visualize.latex_gen as latexlib

tln_lang, hw_tln_lang = tln_spec, mm_tln_spec
spec = mm_tln_spec
IdealV, IdealI = spec.node_type("IdealV"), spec.node_type("IdealI")
IdealE = spec.edge_type("IdealE")
InpV, InpI = spec.node_type("InpV"), spec.node_type("InpI")
MmV, MmI = spec.node_type("MmV"), spec.node_type("MmI")
MmE = spec.edge_type("MmE")
import_fn = {"pulse": pulse_sympy}

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=SympyRewriteGen())

fontsize = 25
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 12,
        "font.family": "Helvetica",
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)


def build_line(graph, e_nt, v_nt, i_nt, length, term_g=1.0, start_i=False):
    tc = 1e-9
    if start_i:
        v_nodes = [v_nt(c=tc, g=0.0) for _ in range(length)] + [v_nt(c=tc, g=term_g)]
        i_nodes = [i_nt(l=tc, r=0.0) for _ in range(length + 1)]
        for i in range(length):
            graph.connect(e_nt(), i_nodes[i], v_nodes[i])
            graph.connect(e_nt(), v_nodes[i], i_nodes[i + 1])
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


def create_tline_branch(
    v_nt: NodeType,
    i_nt: NodeType,
    e_nt: EdgeType,
    e_nt_mm: EdgeType = None,
    mismatch_strategy=None,
    line_len: int = 5,
    branch_len: int = 2,
    branches_per_node: int = 2,
    branch_offset: int = 0,
    branch_stride: int = 1,
):
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse_sympy, g=1.0)
    v_nodes = []
    i_nodes = []
    e_targ_nt = (
        e_nt_mm if mismatch_strategy == "line-only" and e_nt_mm is not None else e_nt
    )
    v_nodes_line, i_nodes_line = build_line(
        graph, e_targ_nt, v_nt, i_nt, line_len, term_g=1.0
    )
    v_nodes += v_nodes_line
    i_nodes += i_nodes_line

    assert line_len % branch_stride == 0
    total_branches = branches_per_node * int(line_len / branch_stride)
    branches = {}
    for i in range(total_branches):
        e_targ_nt = (
            e_nt_mm
            if mismatch_strategy == "branch-only" and e_nt_mm is not None
            else e_nt
        )
        v_nodes_branches, i_nodes_branches = build_line(
            graph, e_targ_nt, v_nt, i_nt, branch_len, start_i=True, term_g=0.0
        )
        branches[i] = (v_nodes_branches, i_nodes_branches)
        v_nodes += v_nodes_branches
        i_nodes += i_nodes_branches

    idx = 0
    for i in range(branch_offset, line_len, branch_stride):
        targ_v = v_nodes_line[i]
        i_nodes_line[i]
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


def create_malformed_tline(
    v_nt: NodeType, i_nt: NodeType, e_nt: EdgeType, line_len=10
) -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse_sympy, g=1.0)
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
    # graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], meas)

    v_nodes[0].name = "IN_V"
    v_nodes[-1].name = "OUT_V"

    return graph, v_nodes, i_nodes


def create_linear_tline(
    v_nt: NodeType, i_nt: NodeType, e_nt: EdgeType, line_len=10
) -> tuple[CDG, list[CDGNode], list[CDGNode]]:
    """Use the given node/edge types to create a single line"""
    spec.reset_type_id()
    graph = CDG()
    current_in = InpI(fn=pulse_sympy, g=1.0)
    v_nodes = [v_nt(c=1e-9, g=0.0) for _ in range(line_len)] + [v_nt(c=1e-9, g=1.0)]
    i_nodes = [i_nt(l=1e-9, r=0.0) for _ in range(line_len)]
    for i in range(line_len):
        graph.connect(e_nt(), v_nodes[i], i_nodes[i])
        graph.connect(e_nt(), i_nodes[i], v_nodes[i + 1])
        graph.connect(IdealE(), v_nodes[i], v_nodes[i])
        graph.connect(IdealE(), i_nodes[i], i_nodes[i])
    graph.connect(IdealE(), v_nodes[-1], v_nodes[-1])

    graph.connect(e_nt(), current_in, v_nodes[0])
    # graph.connect(e_nt(ws=1.0, wt=1.0), v_nodes[-1], meas)

    v_nodes[0].name = "IN_V"
    v_nodes[-1].name = "OUT_V"

    return graph, v_nodes, i_nodes


def nominal_simulation(cdg, time_range, name, post_process_hook=None):
    validator.validate(cdg=cdg, cdg_spec=spec)
    compiler.compile(cdg=cdg, cdg_spec=spec, import_lib=import_fn)
    mapping = compiler.var_mapping
    init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})
    sol = compiler.prog(
        time_range, init_states=init_states, init_seed=123, max_step=1e-10
    )
    time_points = sol.t
    trajs = sol.y
    in_node = list(filter(lambda n: n.name == "IN_V", cdg.nodes))[0]
    out_node = list(filter(lambda n: n.name == "OUT_V", cdg.nodes))[0]
    print(mapping)
    mapping[in_node]
    out_traj_idx = mapping[out_node]

    fig, ax = plt.subplots(1, 1, sharex=True)

    linecolor = "black"
    linewidth = 2.0

    plt.plot(time_points, trajs[out_traj_idx], color=linecolor, linewidth=linewidth)
    ax = plt.gca()
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_xticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(
        r"$\times$10$^{%i}$" % (exponent_axis),
        xy=(1, -0.05),
        xycoords="axes fraction",
        fontsize=fontsize - 5,
    )
    if post_process_hook is not None:
        post_process_hook(fig, ax)

    filename = "gviz-output/tln-example/%s-plot.pdf" % name
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def mismatch_simulation(cdg, time_range, name, post_process_hook=None):
    N_RAND_SIM = 100

    validator.validate(cdg=cdg, cdg_spec=spec)
    compiler.compile(cdg=cdg, cdg_spec=spec, import_lib=import_fn)
    mapping = compiler.var_mapping

    fig, ax = plt.subplots(1, 1, sharex=True)

    init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})
    in_node = list(filter(lambda n: n.name == "IN_V", cdg.nodes))[0]
    out_node = list(filter(lambda n: n.name == "OUT_V", cdg.nodes))[0]
    mapping[in_node]
    out_traj_idx = mapping[out_node]

    alpha = 0.5
    linecolor = "black"
    linewidth = 2.0
    for seed in range(N_RAND_SIM):
        sol = compiler.prog(
            TIME_RANGE, init_states=init_states, init_seed=seed, max_step=1e-10
        )
        time_points = sol.t
        trajs = sol.y
        if seed == 0:
            ax.plot(
                time_points,
                trajs[out_traj_idx],
                alpha=alpha,
                color=linecolor,
                linewidth=linewidth,
            )
        else:
            ax.plot(
                time_points,
                trajs[out_traj_idx],
                alpha=alpha,
                color=linecolor,
                linewidth=linewidth,
            )

    ax = plt.gca()
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_xticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(
        r"$\times$10$^{%i}$" % (exponent_axis),
        xy=(1, -0.05),
        xycoords="axes fraction",
        fontsize=fontsize - 5,
    )

    if post_process_hook is not None:
        post_process_hook(fig, ax)

    filename = "gviz-output/tln-example/%s-plot.pdf" % name
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def graph_process(graph):
    graph.graph.graph_attr["layout"] = "neato"
    graph.graph.graph_attr["sep"] = "+7"
    graph.graph.graph_attr["overlap"] = "false"
    graph.graph.graph_attr["splines"] = "true"


def plot_process(fix, ax):
    pass


def highlight_refl(fig, ax):
    WINDOW_SIZE = 40e-9
    xmin, xmax, ymin, ymax = plt.axis()
    wl = WINDOW_SIZE
    wh = 2 * (WINDOW_SIZE)
    section = np.arange(wl, wh, (wh - wl) * 0.01)
    shadecolor = "yellow"
    shadealpha = 0.2
    np.arange(xmin, xmax, (xmax - xmin) * 0.01)
    ax.axvline(wl, color=shadecolor, lw=2, alpha=shadealpha)
    ax.axvline(wh, color=shadecolor, lw=2, alpha=shadealpha)
    ax.fill_between(section, ymin, ymax, facecolor=shadecolor, alpha=shadealpha)
    ax.fill_between(section, ymin, ymax, facecolor=shadecolor, alpha=shadealpha)


def collapse_derivative(pair: tuple[sp.Symbol, sp.Expr]) -> sp.Eq:
    """Turns tuple of derivative + sympy expression into a single sympy equation."""
    if (var_name := pair[0].name).startswith('ddt_'):
        symbol = sp.symbols(var_name[5:])
        equation = sp.Eq(sp.Derivative(symbol, sp.symbols('time')), pair[1])
        return equation
    else:
        raise ValueError("Not a derivative expression.")




if __name__ == "__main__":
    fnargs = {"br": latexlib.SwitchArg("E_6", "br==1")}
    branch_args = {
        "line_len": 2,
        "branch_stride": 2,
        "branches_per_node": 1,
        "branch_len": 0,
        "branch_offset": 0,
    }
    itl_small_graph, _, _ = create_tline_branch(
        IdealV, IdealI, lambda: IdealE(), **branch_args
    )

    help_fn = []

    sympy_exprs = compiler.compile_sympy(cdg=itl_small_graph, cdg_spec=spec, help_fn=help_fn)
    sympy_eqs = list(map(collapse_derivative, sympy_exprs))
    for eq in sympy_eqs:
        print(eq.subs(sp.symbols('InpI_0_fn'), pulse_sympy))

    print(sympy_eqs)