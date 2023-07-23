"""
Cellular Nonlinear Network (CNN) example.
The template performs linear diffusion for filtering.
- Shows how random mismatch affects the convergence
ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/cta.564
     https://github.com/ankitaggarwal011/PyCNN
"""

from types import FunctionType
import numpy as np
import matplotlib.pyplot as plt
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
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from ark.specification.validation_rule import ValRule, ValPattern
from ark.reduction import SUM


# Ark specification
lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
positive = Range(min=0)

# Ideal implementation
IdealV = NodeType(name='IdealV', order=1,
                  attr_def=[AttrDef('z', attr_type=float)])
Out = NodeType(name='Out', order=0, attr_def=[AttrDef('fn', attr_type=FunctionType)])

# Input should be stateless, setting to 1 just for convenience of setting its value
Inp = NodeType(name='Inp', order=1)
MapE = EdgeType(name='MapE')
FlowE = EdgeType(name='FlowE',
                  attr_def=[AttrDef('g', attr_type=float)])

def saturation(sig):
    """Saturate the value at 1"""
    return 0.5 * (abs(sig + 1) - abs(sig - 1))

# Production rules
Bmat = ProdRule(FlowE, Inp, IdealV, DST, EDGE.g * VAR(SRC))
Dummy = ProdRule(FlowE, Inp, IdealV, SRC, 0) # Dummy rule to make sure Inp is not used
ReadOut = ProdRule(MapE, IdealV, Out, DST, DST.fn(VAR(SRC)))
SelfFeedback = ProdRule(MapE, IdealV, IdealV, SELF, -VAR(SELF) + SELF.z)
Amat = ProdRule(FlowE, Out, IdealV, DST, EDGE.g * VAR(SRC))
prod_rules = [Bmat, Dummy, ReadOut, SelfFeedback, Amat]

# Validation rules
v_val = ValRule(IdealV, [ValPattern(SRC, MapE, Out, Range(exact=1)),
                         ValPattern(DST, FlowE, Out, Range(min=4, max=9)),
                         ValPattern(SELF, FlowE, IdealV, Range(exact=1))])
out_val = ValRule(Out, [ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9)),
                        ValPattern(DST, MapE, IdealV, Range(exact=1))])
inp_val = ValRule(Inp, [ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9))])
val_rules = [v_val, out_val, inp_val]

# Nonideal implementation with 10% random variation

cdg_types = [IdealV, Out, Inp, MapE, FlowE]
help_fn = [saturation]
spec = CDGSpec(cdg_types, prod_rules, val_rules)

validator = ArkValidator(solver=SMTSolver())
compiler = ArkCompiler(rewrite=RewriteGen())

def create_cnn(nrows: int, ncols: int,
               v_nt: NodeType, flow_et: EdgeType,
               A_mat: np.array, B_mat: np.array,
               bias: int) -> CDG:
    """Create a CNN with nrows * ncols nodes
    
    A_mat, B_mat: 3x3 matrices
    bias: bias for the v nodes
    """

    graph = CDG()
    # Create nodes
    vs = [[v_nt(z=bias) for _ in range(ncols)] for _ in range(nrows)]
    inps = [[Inp() for _ in range(ncols)] for _ in range(nrows)]
    outs = [[Out(fn=saturation) for _ in range(ncols)] for _ in range(nrows)]

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

def read_out(nodes, sol, node_2_idx) -> np.array:
    """Read out the solution"""
    nrows, ncols = len(nodes), len(nodes[0])
    traj = np.round(saturation(sol.y.T)).astype(np.uint8)
    imgs = np.zeros((len(traj), nrows, ncols))
    for row_id in tqdm(range(nrows), desc='Read out'):
        for col_id in range(ncols):
            imgs[:, row_id, col_id] = traj[:, node_2_idx[nodes[row_id][col_id]]]
    return imgs


def grayscale_edge_detection(file_name: str):
    """Perform grayscale edge detection on a 96x96 image with cnn"""
    A_mat = np.array([[0.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 0.0]])
    B_mat = np.array([[-1.0, -1.0, -1.0],
                      [-1.0, 8.0, -1.0],
                      [-1.0, -1.0, -1.0]])
    bias = -0.5
    image = rgb_to_gray(plt.imread(file_name))[::4, ::4]
    nrows, ncols = image.shape
    vs, inps, outs, graph = create_cnn(nrows, ncols, IdealV, FlowE, A_mat, B_mat, bias)
    node_mapping = {v: 0 for row in vs for v in row}
    validator.validate(cdg=graph, cdg_spec=spec)
    compiler.compile(graph, spec, help_fn=help_fn, import_lib={}, inline_attr=True, verbose=True)
    node_mapping.update(get_input_mapping(inps, image))
    init_states = compiler.map_init_state(node_mapping)
    sol = compiler.prog([0, 1], init_states=init_states)
    imgs = read_out(vs, sol, compiler.var_mapping)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(imgs[-1], cmap='gray')
    plt.show()


if __name__ == '__main__':
    grayscale_edge_detection('examples/cnn_images/input1.bmp')
