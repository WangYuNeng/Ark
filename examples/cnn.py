"""
Cellular Nonlinear Network (CNN) example.
The template performs linear diffusion for filtering.
- Shows how random mismatch affects the convergence
ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/cta.564
     https://github.com/ankitaggarwal011/PyCNN
"""

from types import FunctionType
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
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
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME, RuleKeyword
from ark.specification.validation_rule import ValRule, ValPattern
from ark.reduction import SUM

# visualization scripts
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.latex_gen_upd as latexlibnew
import ark.visualize.graphviz_gen as graphvizlib


parser = ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    help='Input image')
parser.add_argument('-o', '--output', type=str,
                    help='Output folder')
parser.add_argument('-p', '--paper', action='store_true',
                    help='Generate figures for the paper')
parser.add_argument('-s', '--seed', type=int, default=428,
                    help='Random seed')

args = parser.parse_args()
input_file = args.input
output_folder = args.output
plot_paper = args.paper
seed = args.seed


cnn_lang = CDGLang("cnn")
hw_cnn_lang = CDGLang("hw-cnn", inherits=cnn_lang)
# Ark specification
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
positive = Range(min=0)

Out_vis = NodeType(name='Out', order=0)
SAT = RuleKeyword('sat')
SAT_NI = RuleKeyword('sat_ni')

# Ideal implementation

# Cells in CNN, z is the bias
IdealV = NodeType(name='IdealV', order=1,
                  attr_def=[AttrDef('z', attr_type=float)])

# Outpu function
Out = NodeType(name='Out', order=0, attr_def=[AttrDef('fn', attr_type=FunctionType)])

# Input, should be stateless, setting to 1 just for convenience of setting its value
Inp = NodeType(name='Inp', order=1)
MapE = EdgeType(name='MapE')
FlowE = EdgeType(name='FlowE',
                  attr_def=[AttrDef('g', attr_type=float)])
cnn_lang.add_types(IdealV, Out, Inp)
cnn_lang.add_types(MapE, FlowE)
latexlib.type_spec_to_latex(cnn_lang)


# Nonideal implementation
MmV = NodeType(name='Vm', base=IdealV,
               attr_def=[AttrDefMismatch('mm', attr_type=float, rstd=0.1, attr_range=Range(exact=1))])
MmFlowE_1p = EdgeType(name='fEm_1p', base=FlowE,
                   attr_def=[AttrDefMismatch('g', attr_type=float, rstd=0.01, attr_range=Range(min=-10, max=10))])
MmFlowE_10p = EdgeType(name='fEm', base=FlowE,
                   attr_def=[AttrDefMismatch('g', attr_type=float, rstd=0.1, attr_range=Range(min=-10, max=10))])
OutNL = NodeType(name='OutNL', order=0, base=Out_vis)
hw_cnn_lang.add_types(OutNL)
hw_cnn_lang.add_types(MmV)
hw_cnn_lang.add_types(MmFlowE_10p)
latexlib.type_spec_to_latex(hw_cnn_lang)



def saturation(sig):
    """Saturate the value at 1"""
    return 0.5 * (abs(sig + 1) - abs(sig - 1))

def saturation_diffpair(sig):
    """Saturation function for diffpair implementation"""
    sat_sig = saturation(sig)
    return sat_sig / 0.707107 * np.sqrt(1 - np.square(sat_sig / 2 / 0.707107))

# Production rules
Bmat = ProdRule(FlowE, Inp, IdealV, DST, EDGE.g * VAR(SRC))
Dummy = ProdRule(FlowE, Inp, IdealV, SRC, 0) # Dummy rule to make sure Inp is not used
ReadOut = ProdRule(MapE, IdealV, Out, DST, DST.act(VAR(SRC)))
SelfFeedback = ProdRule(MapE, IdealV, IdealV, SELF, -VAR(SRC) + SRC.z)
Amat = ProdRule(FlowE, Out, IdealV, DST, EDGE.g * VAR(SRC))

ReadOut_vis = ProdRule(MapE, IdealV, Out_vis, DST, SAT.x+(VAR(SRC))) # placeholder to reduce manual work a bit
Amat_vis = ProdRule(FlowE, Out_vis, IdealV, DST, EDGE.g * VAR(SRC))

cnn_lang.add_production_rules(Bmat, ReadOut_vis, SelfFeedback, Amat_vis)


# Production rules for msimatch v
Bmat_mm = ProdRule(FlowE, Inp, MmV, DST, DST.mm * EDGE.g * VAR(SRC))
SelfFeedback_mm = ProdRule(MapE, MmV, MmV, SELF, SRC.mm * (-VAR(SRC) + SRC.z))
Amat_mm = ProdRule(FlowE, Out, MmV, DST, DST.mm * EDGE.g * VAR(SRC))

ReadOut_ni_vis = ProdRule(MapE, IdealV, OutNL, DST, SAT_NI.x+(VAR(SRC))) # placeholder to reduce manual work a bit
Amat_mm_vis = ProdRule(FlowE, Out_vis, MmV, DST, DST.mm * EDGE.g * VAR(SRC))


prod_rules = [Bmat, Dummy, ReadOut, SelfFeedback, Amat, Bmat_mm, SelfFeedback_mm, Amat_mm]
hw_cnn_lang.add_production_rules(Bmat_mm, SelfFeedback_mm, Amat_mm_vis, ReadOut_ni_vis)
latexlib.production_rules_to_latex(hw_cnn_lang)

# Validation rules
v_val = ValRule(IdealV, [ValPattern(SRC, MapE, Out, Range(exact=1)),
                         ValPattern(DST, FlowE, Out, Range(min=4, max=9)),
                         ValPattern(SELF, FlowE, IdealV, Range(exact=1))])
out_val = ValRule(Out, [ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9)),
                        ValPattern(DST, MapE, IdealV, Range(exact=1))])
inp_val = ValRule(Inp, [ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9))])
val_rules = [v_val, out_val, inp_val]
cnn_lang.add_validation_rules(v_val, out_val, inp_val)
latexlib.validation_rules_to_latex(cnn_lang)

latexlibnew.language_to_latex(cnn_lang)
latexlibnew.language_to_latex(hw_cnn_lang)

cdg_types = [IdealV, Out, Inp, MapE, FlowE, MmV, MmFlowE_1p, MmFlowE_10p]
help_fn = [saturation, saturation_diffpair]

spec = CDGSpec(cdg_types, prod_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())

def create_cnn(nrows: int, ncols: int,
               v_nt: NodeType, flow_et: EdgeType,
               A_mat: np.array, B_mat: np.array,
               bias: int, saturation_fn: FunctionType) -> CDG:
    """Create a CNN with nrows * ncols nodes
    
    A_mat, B_mat: 3x3 matrices
    bias: bias for the v nodes
    """

    graph = CDG()
    # Create nodes
    if v_nt == IdealV:
        vs = [[v_nt(z=bias) for _ in range(ncols)] for _ in range(nrows)]
    elif v_nt == MmV:
        vs = [[v_nt(z=bias, mm=1.0) for _ in range(ncols)] for _ in range(nrows)]
    inps = [[Inp() for _ in range(ncols)] for _ in range(nrows)]
    outs = [[Out(act=saturation_fn) for _ in range(ncols)] for _ in range(nrows)]

    # Create edges
    # All v nodes connect from self, and connect to output
    # in/output node in the corner -> connect the v node in that position and 3 neighbors v nodes
    # in/output node on the edge -> connect the v node in that position and 5 neighbors v nodes
    # in/output node in the middle -> connect the v node in that position and 8 neighbors v nodes
    for row_id in range(nrows):
        for col_id in range(ncols):
            v = vs[row_id][col_id]
            inp = inps[row_id][col_id]
            out = outs[row_id][col_id]
            graph.connect(MapE(), v, v)
            graph.connect(MapE(), v, out)

            for row_offset in [-1, 0, 1]:
                for col_offset in [-1, 0, 1]:
                    if row_id + row_offset < 0 or row_id + row_offset >= nrows:
                        continue
                    if col_id + col_offset < 0 or col_id + col_offset >= ncols:
                        continue
                    graph.connect(flow_et(g=B_mat[row_offset + 1, col_offset + 1]),
                                  inp, vs[row_id + row_offset][col_id + col_offset])
                    graph.connect(flow_et(g=A_mat[row_offset + 1, col_offset + 1]),
                                  out, vs[row_id + row_offset][col_id + col_offset])
    return vs, inps, outs, graph

def rgb_to_gray(rgb):
    """Convert rgb image to grayscale"""
    return np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])).astype(np.uint8)

def get_input_mapping(inps, image) -> dict[CDGNode, float]:
    """Set the input mapping for the input nodes"""
    nrows, ncols = image.shape
    mapping = {}
    for row_id in range(nrows):
        for col_id in range(ncols):
            mapping[inps[row_id][col_id]] = image[row_id, col_id]
    return mapping

def read_out(nodes, sol, node_2_idx, time_points, saturation_fn) -> np.array:
    """Read out the solution"""
    nrows, ncols = len(nodes), len(nodes[0])
    traj = np.round(saturation_fn(sol.sol(time_points).T)).astype(np.uint8)
    imgs = np.zeros((len(traj), nrows, ncols))
    for row_id in tqdm(range(nrows), desc='Read out'):
        for col_id in range(ncols):
            imgs[:, row_id, col_id] = traj[:, node_2_idx[nodes[row_id][col_id]]]
    return imgs

def prepare_tpl():
    """Edge detection template"""
    A_mat = np.array([[0.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 0.0]])
    B_mat = np.array([[-1.0, -1.0, -1.0],
                      [-1.0, 8.0, -1.0],
                      [-1.0, -1.0, -1.0]])
    bias = -0.5
    return A_mat, B_mat, bias

def layout_dense_graph(graph):
    graph.graph.graph_attr["layout"] = "neato"
    graph.graph.graph_attr["sep"] = "+7"
    graph.graph.graph_attr["overlap"] = "false"
    graph.graph.graph_attr["splines"] = "true"


def sim_cnn(image: np.array,
            A_mat: np.array, B_mat: np.array, bias: int,
            time_points: np.array,
            v_nt: NodeType, flow_et: EdgeType,
            saturation_fn: FunctionType, index=0):
    """Simulate the cnn with ARK"""
    nrows, ncols = image.shape
    vs, inps, outs, graph = create_cnn(nrows, ncols, v_nt, flow_et,
                                               A_mat, B_mat, bias, saturation_fn)

    # _, _, _, small_graph = create_cnn(3, 3, v_nt, flow_et,
    #                                            A_mat, B_mat, bias, saturation_fn)

    # print("index = %d" % index)
    # graphvizlib.cdg_to_graphviz("cnn", "cnn_inh_%d" % index , hw_cnn_lang,small_graph,inherited=True, post_layout_hook=layout_dense_graph)
    # graphvizlib.cdg_to_graphviz("cnn", "cnn_%d" % index , hw_cnn_lang,small_graph,inherited=False,post_layout_hook=layout_dense_graph)

    node_mapping = {v: 0 for row in vs for v in row}
    validator.validate(cdg=graph, cdg_spec=spec)
    if flow_et == FlowE and v_nt == IdealV:
        compiler.compile(graph, spec, help_fn=help_fn, import_lib={},
                        inline_attr=True, verbose=True)
    else:
        compiler.compile(graph, spec, help_fn=help_fn, import_lib={},
                        inline_attr=False, verbose=True)
    node_mapping.update(get_input_mapping(inps, image))
    init_states = compiler.map_init_state(node_mapping)
    sol = compiler.prog([0, 1], init_states=init_states, dense_output=True)
    imgs = read_out(vs, sol, compiler.var_mapping,
                            time_points, saturation_fn)
    return imgs

def grayscale_edge_detection(file_name: str):
    """Perform grayscale edge detection on a image with cnn"""
    A_mat, B_mat, bias = prepare_tpl()
    image = rgb_to_gray(plt.imread(file_name))
    nrows, ncols = image.shape

    TIME_RANGE = [0, 1]
    time_points = np.linspace(*TIME_RANGE, 9)[1:]

    nt_list = [MmV, IdealV, IdealV, IdealV]
    et_list = [FlowE, FlowE, MmFlowE_1p, MmFlowE_10p]
    idx = 0
    for saturation_fn in [saturation, saturation_diffpair]:
        for v_nt, flow_et in zip(nt_list, et_list):
            imgs = sim_cnn(image, A_mat, B_mat, bias, time_points,
                            v_nt, flow_et, saturation_fn, index=idx)

            fig, axs = plt.subplots(3, 3)
            axs[0, 0].imshow(image, cmap='gray')
            axs[0, 0].set_title('Input img')
            for i, (time, img) in enumerate(zip(time_points, imgs)):
                ax = axs[(i + 1) // 3, (i + 1) % 3]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'time: {time}')
            title = f'{v_nt.name}_{flow_et.name}'
            plt.suptitle(title)
            plt.tight_layout()
            save_path = os.path.join(output_folder, f'{saturation_fn.__name__}_{title}.png')
            plt.savefig(save_path)
            idx += 1

def paper_plot():
    """Generate the plots for the paper"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 12,
        "font.family": "Helvetica",
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30
    })
    col_titles = [r'\texttt{A}', r'\texttt{B}', r'\texttt{C}', r'\texttt{D}']

    # plot saturation and saturation_diffpair in [-1.5, 1.5] in the same figure
    x = np.linspace(-1.2, 1.2, 100)
    # plt.plot(x, saturation(x), label='sat-ideal', linewidth=5.0)
    plt.plot(x, saturation(x), linewidth=5.0)
    # plt.plot(x, saturation_diffpair(x), label='sat-diffpair',linewidth=5.0)
    plt.plot(x, saturation_diffpair(x),linewidth=5.0)
    # plt.legend(fontsize=20)
    # plt.xlabel('input', fontsize=15)
    # plt.ylabel('output', fontsize=15)
    # plt.grid()
    plt.savefig('examples/cnn_images/saturation-cmp.pdf',bbox_inches='tight')
    plt.close()


    components = [[IdealV, FlowE, saturation], [MmV, FlowE, saturation],
                  [IdealV, MmFlowE_10p, saturation],[IdealV, FlowE, saturation_diffpair]]

    A_mat, B_mat, bias = prepare_tpl()
    image = rgb_to_gray(plt.imread('examples/cnn_images/cnn_input.png'))
    N_ROW = 5
    TIME_RANGE = [0, 1]
    time_points = np.linspace(*TIME_RANGE,N_ROW)
    fig, axs = plt.subplots(N_ROW, len(components), figsize=(4.5, 4.8))
    fig.subplots_adjust(wspace=0, hspace=0)
    for j, (col_title, (v_nt, flow_et, saturation_fn)) in enumerate(zip(col_titles, components)):
        imgs = sim_cnn(image, A_mat, B_mat, bias, time_points,
                        v_nt, flow_et, saturation_fn)
        for i, (time, img) in enumerate(zip(time_points, imgs)):
            ax = axs[i, j]
            # remove axis ticks but keep the labels
            ax.tick_params(axis='both', which='both', bottom=False, top=False,
                            labelbottom=False, right=False, left=False, labelleft=False)
            ax.imshow(img, cmap='gray')
            if i == 0:
                ax.set_title(col_title)
            if j == 0:
                ax.set_ylabel(f't={time:.2f}', rotation=0, labelpad=20)

    # put rows in plt closer together
    plt.savefig('examples/cnn_images/cnn-output.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    np.random.seed(seed)
    if not plot_paper:
        grayscale_edge_detection(input_file)
    else:
        paper_plot()
