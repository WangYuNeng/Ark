"""
CNN for grayscale edge detection
- Shows how random mismatch affects the convergence
ref: https://github.com/ankitaggarwal011/PyCNN
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
from ark.cdg.cdg import CDG, CDGNode
from ark.specification.cdg_types import NodeType, EdgeType
from spec import mm_cnn_spec, saturation, saturation_diffpair


cnn_spec = mm_cnn_spec
help_fn = [saturation, saturation_diffpair]
IdealV = cnn_spec.node_type('IdealV')
Out, Inp = cnn_spec.node_type('Out'), cnn_spec.node_type('Inp')
MapE, FlowE = cnn_spec.edge_type('MapE'), cnn_spec.edge_type('FlowE')
Vm = cnn_spec.node_type('Vm')
fEm_1p, fEm_10p = cnn_spec.edge_type('fEm_1p'), cnn_spec.edge_type('fEm_10p')

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
    elif v_nt == Vm:
        vs = [[v_nt(z=bias, mm=1.0) for _ in range(ncols)] for _ in range(nrows)]
    inps = [[Inp() for _ in range(ncols)] for _ in range(nrows)]
    outs = [[Out(act=saturation_fn) for _ in range(ncols)] for _ in range(nrows)]

    # Create edges
    # All v nodes connect from self, and connect to output
    # in/output node in the corner 
    # -> connect v node in that position and 3 neighbors v nodes
    # in/output node on the edge 
    # -> connect the v node in that position and 5 neighbors v nodes
    # in/output node in the middle 
    # -> connect the v node in that position and 8 neighbors v nodes
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

def sim_cnn(image: np.array,
            A_mat: np.array, B_mat: np.array, bias: int,
            time_points: np.array,
            v_nt: NodeType, flow_et: EdgeType,
            saturation_fn: FunctionType, index=0):
    """Simulate the cnn with ARK"""
    nrows, ncols = image.shape
    vs, inps, outs, graph = create_cnn(nrows, ncols, v_nt, flow_et,
                                               A_mat, B_mat, bias, saturation_fn)

    node_mapping = {v: 0 for row in vs for v in row}
    validator.validate(cdg=graph, cdg_spec=cnn_spec)
    if flow_et == FlowE and v_nt == IdealV:
        compiler.compile(graph, cnn_spec, help_fn=help_fn, import_lib={},
                        inline_attr=True, verbose=True)
    else:
        compiler.compile(graph, cnn_spec, help_fn=help_fn, import_lib={},
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

    nt_list = [Vm, IdealV, IdealV, IdealV]
    et_list = [FlowE, FlowE, fEm_1p, fEm_10p]
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
            save_path = os.path.join(output_folder,
                                     f'{saturation_fn.__name__}_{title}.png')
            plt.savefig(save_path)
            idx += 1

def paper_plot():
    """Generate the plots for the paper"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 12,
        "font.family": "Helvetica",
    })
    col_titles = [r'\texttt{A}', r'\texttt{B}', r'\texttt{C}', r'\texttt{D}']

    x = np.linspace(-1.2, 1.2, 100)
    plt.plot(x, saturation(x), linewidth=5.0)
    plt.plot(x, saturation_diffpair(x),linewidth=5.0)
    plt.savefig('saturation-cmp.pdf',bbox_inches='tight')
    plt.close()


    components = [[IdealV, FlowE, saturation], [Vm, FlowE, saturation],
                  [IdealV, fEm_10p, saturation],[IdealV, FlowE, saturation_diffpair]]

    A_mat, B_mat, bias = prepare_tpl()
    image = rgb_to_gray(plt.imread('cnn_input.png'))
    N_ROW = 5
    TIME_RANGE = [0, 1]
    time_points = np.linspace(*TIME_RANGE,N_ROW)
    fig, axs = plt.subplots(N_ROW, len(components), figsize=(4.5, 4.8))
    fig.subplots_adjust(wspace=0, hspace=0)
    for j, (col_title, (v_nt, flow_et, saturation_fn)) \
        in enumerate(zip(col_titles, components)):
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
    plt.savefig('cnn-output.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

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

    np.random.seed(seed)
    if not plot_paper:
        grayscale_edge_detection(input_file)
    else:
        paper_plot()
