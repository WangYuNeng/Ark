"""
Example: Coupled Oscillator Network
[1] https://www.nature.com/articles/s41598-019-49699-5
[2] https://ieeexplore.ieee.org/document/9531734

Issue: Can't reproduce the results in [1] and [2]. The solutions are not
100% correct even in the very simple 4 nodes cases. Some reasons:
1. I fail to find some parameters, i.e., A0 in [1] and noise current in [2]
2. It turns outh transient noise is somewhat necessary to explore the search
   space as described in [2]. However, transient noise is not well supported
   in python simulation. I tried some toy examples and it doesn't seem to work
   either.

Taking a step back, showing the distortion of the sine function in [2] can
affect the results for now. Weirdly enough, the results get better with disctortion
"""

from types import FunctionType
from itertools import product
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME

F_1 = 50e3
T_1 = 1 / F_1
A0, TAU = F_1, 5 * T_1

F_2 = 795.8e6
T_2 = 1 / F_2

NOIS_STD = 100

N_TIRAL = 1

Osc = NodeType(name='Osc', order=1, attr_def=[AttrDef('lock_fn', attr_type=FunctionType),
                                               AttrDef('osc_fn', attr_type=FunctionType),
                                               AttrDef('noise_fn', attr_type=FunctionType)])
Coupling = EdgeType(name='Coupling', attr_def=[AttrDef('k', attr_type=float)])

Osc1 = NodeType(name='Osc1', base=Osc)
Osc2 = NodeType(name='Osc2', base=Osc)
Coupling_distorted = EdgeType(name='Coupling_distorted', base=Coupling,
                              attr_def=[AttrDefMismatch('offset', attr_type=float, rstd=0.1),
                                        AttrDefMismatch('scale', attr_type=float, rstd=0.1)])

def locking_fn_1(t, x, a0, tau):
    """Injection locking function from [1]"""
    return a0 * np.exp(-t / tau) * np.sin(2 * x)

def locking_fn_2(x):
    """Injection locking function from [2]"""
    return 2 * 795.8e6 * np.sin(2 * np.pi * x)

def coupling_fn_1(x):
    """Coupling function from [1]"""
    return np.sin(x)

def coupling_fn_2(x):
    """Coupling function from [2]"""
    return 2 * 795.8e6 * np.sin(np.pi * x)

def normal_noise(std):
    """Normal noise"""
    return np.random.normal(0, std)

def zero_noise(_):
    """Zero noise"""
    return 0

r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, - EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST)))
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, - EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC)))
r_lock_1 = ProdRule(Coupling, Osc1, Osc1, SELF,
                    - SELF.lock_fn(TIME, VAR(SELF), A0, TAU) - SELF.noise_fn(NOIS_STD))
r_lock_2 = ProdRule(Coupling, Osc2, Osc2, SELF,
                    - SELF.lock_fn(VAR(SELF)) - SELF.noise_fn(NOIS_STD))
r_cp_src_distorted = ProdRule(Coupling_distorted, Osc2, Osc2, SRC, 
                              - EDGE.k * (EDGE.scale * SRC.osc_fn(VAR(SRC) - VAR(DST)) + EDGE.offset))
r_cp_dst_distorted = ProdRule(Coupling_distorted, Osc2, Osc2, DST, 
                              - EDGE.k * (EDGE.scale * SRC.osc_fn(VAR(DST) - VAR(SRC)) + EDGE.offset))


cdg_types = [Osc, Coupling, Osc1, Osc2]
production_rules = [r_cp_src, r_cp_dst, r_lock_1, r_lock_2, r_cp_src_distorted, r_cp_dst_distorted]
help_fn = [locking_fn_1, locking_fn_2, coupling_fn_1, coupling_fn_2, normal_noise, zero_noise]

def create_max_cut_con(connection_mat, osc_nt: NodeType, cp_et: EdgeType, noise_fn: FunctionType):
    """Create a CDG of con for solving MAXCUT of the graph described by connection_mat"""
    if osc_nt == Osc1:
        locking_fn = locking_fn_1
        cp_fn = coupling_fn_1
    elif osc_nt == Osc2:
        locking_fn = locking_fn_2
        cp_fn = coupling_fn_2
    nodes = [osc_nt(lock_fn=locking_fn, osc_fn=cp_fn, noise_fn=noise_fn) for _ in range(4)]
    if cp_et == Coupling:
        args = {'k': -1.0}
    elif cp_et == Coupling_distorted:
        args = {'k': -1.0, 'offset': 0.0, 'scale': 1.0}

    graph = CDG()
    for i, row in enumerate(connection_mat):
        for j, val in enumerate(row):
            if val:
                graph.connect(cp_et(**args), nodes[i], nodes[j])
    for i in range(4):
        graph.connect(Coupling(k=1.0), nodes[i], nodes[i])

    return nodes, graph

def calc_cut_size(connection_mat, cut):
    """Calculate the cut size of a cut"""
    cut_val = 0
    for i, ass0 in enumerate(cut):
        for j, ass1 in enumerate(cut[i+1:]):
            # If the nodes are in different subsets, add the edge value to the cut value
            if ass0 != ass1:
                cut_val += connection_mat[i][i + 1 + j]
    return cut_val

def gen_max_cut_prob(seed):
    """Generate a max cut problem"""
    np.random.seed(seed)
    connection_mat = np.random.randint(0, 2, size=(4, 4))
    # set diagonal and lower triangle to 0
    connection_mat = np.triu(connection_mat, 1)

    cut_enumeration = [[a, b, c, 0] for a, b, c in product([0, 1], repeat=3)]
    max_cut = 0
    for cut in cut_enumeration:
        cut_val = calc_cut_size(connection_mat, cut)
        max_cut = max(max_cut, cut_val)
    return connection_mat, max_cut

def plot_oscillation(time_points, sol, mapping, omega, scaling, title=None):
    """Plot the oscillation of the oscillator"""
    fig, ax = plt.subplots(nrows=2)
    for node, idx in mapping.items():
        phi = sol.sol(time_points)[idx].T * scaling
        ax[1].plot(time_points,
                np.sin(omega * time_points + phi),
                label=node.name)
        ax[0].plot(time_points, phi)
    ax[0].set_title('phase (phi)')
    ax[1].set_title('sin(wt + phi)')
    ax[0].legend(), ax[1].legend()
    ax[-1].set_xlabel('time')
    plt.tight_layout()
    if title:
        plt.savefig(title)
    plt.show()


def main():

    problems = [gen_max_cut_prob(seed) for seed in range(N_TIRAL)]
    # print(problems[:3])
    test_prob = np.array([[0, 1, 1, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
    problems = [(test_prob, 3) for _ in range(N_TIRAL)]

    print('Solving success rate')
    for osc_nodetype in [Osc1, Osc2]:
        if osc_nodetype == Osc1:
            cycle = T_1
            omega = 2 * np.pi * F_1
            scaling = 1
        elif osc_nodetype == Osc2:
            cycle = T_2
            omega = 2 * np.pi * F_2
            scaling = np.pi

        for cp_et in [Coupling, Coupling_distorted]:
            if cp_et == Coupling_distorted and osc_nodetype == Osc1:
                continue
            for noise_fn in [zero_noise, normal_noise]:
                success = 0
                for seed, prob in tqdm(enumerate(problems),
                                       desc=f'{osc_nodetype.name}/{cp_et.name}/{noise_fn.__name__}',
                                       total=N_TIRAL):
                    connection_mat, max_cut = prob
                    nodes, graph = create_max_cut_con(connection_mat, osc_nodetype, cp_et, noise_fn)
                    spec = CDGSpec(cdg_types, production_rules, None)
                    compiler = ArkCompiler(rewrite=RewriteGen())
                    compiler.compile(cdg=graph, cdg_spec=spec, help_fn=help_fn, import_lib={})
                    compiler.dump_prog(f'{osc_nodetype.name}{cp_et.name}{noise_fn.__name__}.py')
                    time_range = [0, 5 * cycle]
                    time_points = np.linspace(*time_range, 1000)
                    mapping = compiler.var_mapping
                    np.random.seed(seed)
                    init_states = compiler.map_init_state({node: np.random.rand() * np.pi / scaling
                                                    for node in mapping.keys()})
                    sol = compiler.prog(time_range, init_states=init_states,
                                        sim_seed=seed, dense_output=True)
                    phase = {}
                    for node, idx in mapping.items():
                        phi = sol.sol(time_points)[idx].T * scaling
                        phase[node] = np.round(phi[-1] / np.pi) % 2
                    assigment = [phase[node] for node in nodes]
                    cut_size = calc_cut_size(connection_mat, assigment)
                    if cut_size == max_cut:
                        success += 1
                print(f'{osc_nodetype.name}/{cp_et.name}/{noise_fn.__name__}: {success / N_TIRAL * 100}%')

    

if __name__ == '__main__':
    main()
