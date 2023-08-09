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

Parameters:
F_1, T_1, TAU are from [1]. I can't find the value of A0 in the paper.
The value is hand-tuned to make the oscillator to settle within 5 cycles.

F_2, T_2 are from [2].

NOIS_STD is the standard deviation of the noise current.
The value is set quite arbitrarily.

N_TRIAL is the number of trials to run the simulation.
ATOL, RTOL are the absolute and relative tolerance to determine whether the oscillators
sync up successfully. The tolerance will affect whether we consider a problem is solved or not.
For example, setting the coupling to have higher offset will make the synchronization imperfect.
If the tolerance is high, the problem will be considered solved even though you can see small phase
difference in the plot.

Some observations:
1.  The system is sensitive to parameters. Can't find the good params for Osc1.
    It is no better than random guess now if compare
        python examples/con.py --baseline
        with
        python examples/con.py --osc_type 1
    Therefore, we focus on Osc2 for now.
2.  The system is quite resilient to random scaling of coupling sine functions.
    It even solves slightly more problems than nominal cases.
    E.g., compare
        python examples/con.py
        with
        python examples/con.py --scale_rstd 0.1
    It's not universally better though. For example, there are cases that solved by
    the nominal system but not by the scaled one.
3.  Offset could cause a problem in synchronization. For example, if run
        python examples/con.py --offset_rstd 0.1
    The synchronization success rate drops significantly (92.3% -> 5.1%)
    However, the offset mismatch helps get out of local minimum if we allow slightly
    tolerance in the definition of "synchronized". For example, if we initialize the
    states in local minimum, the nominal system and scaled system won't get out of it.
    E.g., run
        python examples/con.py --initialize 1
        python examples/con.py --initialize 1 --scale_rstd 0.1
        The phase stuck at the initial value and not solving the problem.
        (add -p to plot the phase for visualization)
    However, with offset mismatch, the system can get out of the local minimum sometimes.
    E.g., run
        python examples/con.py --initialize 1 --offset_rstd 0.1
    can have a better correct rate (1% -> 10.9%).
    If we further increase the synchronization tolerance, it would solve more.
    E.g., run
        python examples/con.py --initialize 1 --offset_rstd 0.1 --atol 0.1 --rtol 0.1
    The correct rate becomes 61%.
"""

from types import FunctionType
from argparse import ArgumentParser
from itertools import product
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
# from ark.solver import SMTSolver
# from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME, RuleKeyword
from ark.specification.range import Range

# visualization scripts
from ark.cdg.cdg_lang import CDGLang
import ark.visualize.latex_gen as latexlib
import ark.visualize.latex_gen_upd as latexlibnew
import ark.visualize.graphviz_gen as graphvizlib

parser = ArgumentParser()
parser.add_argument('--osc_type', type=int, default=2)
parser.add_argument('--baseline', action='store_true')
parser.add_argument('-p', '--plot', action='store_true')
parser.add_argument('-n', '--n_trial', type=int, default=1000)
parser.add_argument('--n_cycle', type=int, default=5, help='Number of cycles to run the simulation')
parser.add_argument('--atol', type=float, default=1e-2)
parser.add_argument('--rtol', type=float, default=1e-2)
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--offset_rstd', type=float, default=None,
                    help='Standard deviation of the offset relative to the sine function. \
                          The disctibution is N(0, offset_rstd * 2 * F_2) \
                          The 2 * F_2 term is to make the mismatch same order as scale_rstd.')
parser.add_argument('--scale_rstd', type=float, default=None,
                    help='Standard deviation of the scaling relative to the coupling function. \
                          The disctibution is N(1, scale_rstd) * coupling_fn.')
parser.add_argument('--initialize', type=float, default=None)
parser.add_argument('--w_n_bits', type=int, default=1, help='Number of bits for cut weights')
sim_args = parser.parse_args()

F_1 = 50e3
T_1 = 1 / F_1
A0, TAU =  2 * F_1, 5 * T_1

F_2 = 795.8e6
T_2 = 1 / F_2

NOIS_STD = 100

SIM_DISTORTED = False

OSC_TYPE = sim_args.osc_type
BASELINE = sim_args.baseline
N_CYCLE = sim_args.n_cycle
PLOT = sim_args.plot
N_TIRAL = sim_args.n_trial
ATOL, RTOL = sim_args.atol, sim_args.rtol
NOISE = sim_args.noise
OFFSET_RSTD = sim_args.offset_rstd
SCALE_RSTD = sim_args.scale_rstd
INITIALIZE = sim_args.initialize
W_N_BITS = sim_args.w_n_bits

if OSC_TYPE == 1 and (OFFSET_RSTD or SCALE_RSTD):
    raise ValueError('Osc1 does not support OFFSET or SCALE')

obc_lang = CDGLang("obc")

Osc = NodeType(name='Osc', order=1, attr_def=[AttrDef('lock_fn', attr_type=FunctionType, nargs=1),
                                               AttrDef('osc_fn', attr_type=FunctionType, nargs=1)])
                                            #    AttrDef('noise_fn', attr_type=FunctionType)])
Coupling = EdgeType(name='Cpl', attr_def=[AttrDef('k', attr_type=float, attr_range=Range(min=-8, max=8))])

Osc1 = NodeType(name='Osc1', base=Osc)
Osc2 = NodeType(name='Osc2', base=Osc)

Osc_vis = NodeType(name='Osc', order=1)

obc_lang.add_types(Osc_vis,Coupling)
latexlib.type_spec_to_latex(obc_lang)

Coupling_distorted = None
if OFFSET_RSTD and SCALE_RSTD:
    offset_std = OFFSET_RSTD * 2 * F_2
    Coupling_distorted = EdgeType(name='Coupling_distorted', base=Coupling,
                                  attr_def=[AttrDefMismatch('offset', attr_type=float,
                                                            std=offset_std),
                                            AttrDefMismatch('scale', attr_type=float,
                                                            rstd=SCALE_RSTD)])
elif OFFSET_RSTD:
    offset_std = OFFSET_RSTD * 2 * F_2
    offset_std_norm = OFFSET_RSTD * 2 * F_2 / (1.2 * F_2)
    Coupling_distorted = EdgeType(name='Cpl_ofs', base=Coupling,
                                  attr_def=[AttrDefMismatch('offset', attr_type=float,
                                                            std=offset_std_norm, attr_range=Range(exact=0))])
                                            # AttrDef('scale', attr_type=float)])
elif SCALE_RSTD:
    Coupling_distorted = EdgeType(name='Coupling_distorted', base=Coupling,
                                  attr_def=[AttrDef('offset', attr_type=float),
                                            AttrDefMismatch('scale', attr_type=float,
                                                            rstd=SCALE_RSTD)])
                                                        
if not Coupling_distorted is None:
    hw_obc_lang = CDGLang("ofs-obc", inherits=obc_lang)
    hw_obc_lang.add_types(Coupling_distorted)
    latexlib.type_spec_to_latex(hw_obc_lang)



def locking_fn_1(t, x, a0, tau):
    """Injection locking function from [1]"""
    return a0 * (1 - np.exp(-t / tau)) * np.sin(2 * x)

def locking_fn_2(x):
    """Injection locking function from [2]
    Modify the leading coefficient to 1.2 has a better outcome
    """
    return 1.2 * 795.8e6 * np.sin(2 * np.pi * x)

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
r_lock = ProdRule(Coupling, Osc, Osc, SELF, - SRC.lock_fn(VAR(SRC)))

SIN = RuleKeyword('sin')
SIN2 = RuleKeyword('sin2')

# placeholder to reduce manual work a bit
r_cp_src_vis = ProdRule(Coupling, Osc_vis, Osc_vis, SRC, - EDGE.k * SIN.x * (VAR(SRC) - VAR(DST)))
r_cp_dst_vis = ProdRule(Coupling, Osc_vis, Osc_vis, DST, - EDGE.k * SIN.x * (VAR(DST) - VAR(SRC)))
r_lock_vis = ProdRule(Coupling, Osc_vis, Osc_vis, SELF, - SIN2.x * (VAR(SRC)))

r_lock_1 = ProdRule(Coupling, Osc1, Osc1, SELF,
                    - SRC.lock_fn(TIME, VAR(SRC), A0, TAU) - SRC.noise_fn(NOIS_STD))
r_lock_2 = ProdRule(Coupling, Osc2, Osc2, SELF,
                    - SRC.lock_fn(VAR(SRC)))
                    # - SRC.lock_fn(VAR(SRC)) - SRC.noise_fn(NOIS_STD))
obc_lang.add_production_rules(r_cp_src_vis,r_cp_dst_vis, r_lock_vis)

cdg_types = [Osc, Coupling, Osc1, Osc2]
production_rules = [r_cp_src, r_cp_dst, r_lock_1, r_lock_2]
help_fn = [locking_fn_1, locking_fn_2, coupling_fn_1, coupling_fn_2, normal_noise, zero_noise]

if Coupling_distorted:
    SIM_DISTORTED = True
    r_cp_src_distorted = ProdRule(Coupling_distorted, Osc2, Osc2, SRC,
                              - EDGE.k * (EDGE.scale * SRC.osc_fn(VAR(SRC) - VAR(DST)) \
                                           + EDGE.offset))
    r_cp_dst_distorted = ProdRule(Coupling_distorted, Osc2, Osc2, DST,
                              - EDGE.k * (EDGE.scale * SRC.osc_fn(VAR(DST) - VAR(SRC)) \
                                          + EDGE.offset))
    production_rules += [r_cp_src_distorted, r_cp_dst_distorted]

    r_cp_src_distorted_vis = ProdRule(Coupling_distorted, Osc, Osc, SRC,
                              - EDGE.k * (SIN.x * (VAR(SRC) - VAR(DST)) \
                                           + EDGE.offset))
    r_cp_dst_distorted_vis = ProdRule(Coupling_distorted, Osc, Osc, DST,
                              - EDGE.k * (SIN.x * (VAR(DST) - VAR(SRC)) \
                                          + EDGE.offset))
    hw_obc_lang.add_production_rules(r_cp_src_distorted_vis, r_cp_dst_distorted_vis)


latexlib.production_rules_to_latex(obc_lang)
latexlibnew.language_to_latex(obc_lang)
if not Coupling_distorted is None:
    latexlib.production_rules_to_latex(hw_obc_lang)
    latexlibnew.language_to_latex(hw_obc_lang)


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
                args['k'] = float(-val)
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

def gen_max_cut_prob(seed, w_n_bits=W_N_BITS):
    """Generate a max cut problem"""
    np.random.seed(seed)
    connection_mat = np.random.randint(0, 2 ** w_n_bits, size=(4, 4))
    # set diagonal and lower triangle to 0
    connection_mat = np.triu(connection_mat, 1)

    cut_enumeration = [[a, b, c, 0] for a, b, c in product([0, 1], repeat=3)]
    max_cut, max_cut_size = [0 for _ in range(4)], 0
    for cut in cut_enumeration:
        cut_val = calc_cut_size(connection_mat, cut)
        if cut_val > max_cut_size:
            max_cut, max_cut_size = cut, cut_val
    return connection_mat, max_cut_size, max_cut


def plot_oscillation(time_points, sol, mapping, omega, scaling, title=None):
    """Plot the oscillation of the oscillator"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 15,
        "font.family": "Helvetica"
    })
    cycle = time_points / T_2
    fig, ax = plt.subplots(nrows=2)
    for node, idx in mapping.items():
        phi = sol.sol(time_points)[idx].T * scaling
        ax[1].plot(cycle,
                np.sin(omega * time_points + phi),
                label=node.name)
        ax[0].plot(cycle, phi)
    ax[0].set_title(r'$\phi$')
    ax[0].set_ylabel('phase (rad)')
    ax[1].set_title(r'$\sin{(\omega t + \phi)}$')
    ax[1].set_ylabel('Amplitude (V)')
    ax[1].set_xlabel('\# of cycle (t/T)')
    plt.tight_layout()
    if title:
        plt.savefig(title+'.pdf')
    plt.show()

def phase_to_assignment(phase, atol=ATOL, rtol=RTOL):
    """Convert a phase to an assignment"""
    n_pi = np.round(phase / np.pi)
    phase_reconstruct = n_pi * np.pi
    if np.allclose(phase, phase_reconstruct, atol=atol, rtol=rtol):
        return n_pi % 2
    return None

def main():

    problems = [gen_max_cut_prob(seed) for seed in range(N_TIRAL)]

    if BASELINE:
        correct = 0
        for prob in tqdm(problems,
                         desc='Baseline (random guess)',
                         total=N_TIRAL):
            connection_mat, max_cut = prob
            assignments = np.random.randint(0, 2, size=4)
            cut_size = calc_cut_size(connection_mat, assignments)
            if cut_size == max_cut:
                correct += 1
        print(f'Correct rate: {correct / N_TIRAL * 100}%')
        exit()

    if SIM_DISTORTED:
        cp_et = Coupling_distorted
    else:
        cp_et = Coupling

    if OSC_TYPE == 1:
        osc_nodetype = Osc1
        cycle = T_1
        omega = 2 * np.pi * F_1
        scaling = 1
    elif OSC_TYPE == 2:
        osc_nodetype = Osc2
        cycle = T_2
        omega = 2 * np.pi * F_2
        scaling = np.pi

    if not NOISE:
        noise_fn = zero_noise
    else:
        noise_fn = normal_noise

    correct = 0
    sync_success = 0
    for seed, prob in tqdm(enumerate(problems),
                            desc=f'{osc_nodetype.name}, {cp_et.name}, {noise_fn.__name__}',
                            total=N_TIRAL):
        connection_mat, max_cut_size, max_cut = prob
        nodes, graph = create_max_cut_con(connection_mat, osc_nodetype, cp_et, noise_fn)

        lang = obc_lang if Coupling_distorted is None else hw_obc_lang
        if seed == 0:
            graphvizlib.cdg_to_graphviz("con", "con_%d_inh" % seed, lang,graph,inherited=True)
            graphvizlib.cdg_to_graphviz("con", "con_%d" % seed, lang,graph,inherited=False)
        
        spec = CDGSpec(cdg_types, production_rules, None)
        compiler = ArkCompiler(rewrite=RewriteGen())
        compiler.compile(cdg=graph, cdg_spec=spec, help_fn=help_fn, import_lib={})
        time_range = [0, N_CYCLE * cycle]
        time_points = np.linspace(*time_range, 1000)
        mapping = compiler.var_mapping
        if INITIALIZE is not None:
            init_states = compiler.map_init_state({node: INITIALIZE for node in mapping.keys()})
        else:
            np.random.seed(seed)
            init_states = compiler.map_init_state({node: np.random.rand() * np.pi / scaling
                                                   for node in mapping.keys()})
        sol = compiler.prog(time_range, init_states=init_states,
                            sim_seed=seed, dense_output=True)
        if seed == 1 and PLOT:
            plot_oscillation(time_points, sol, mapping, omega, scaling,
                                title=f'{osc_nodetype.name}, {cp_et.name}'
                             )
        node_to_assignment = {}
        sync_failed = False
        for node, idx in mapping.items():
            phi = sol.sol(time_points)[idx].T * scaling
            assigment = phase_to_assignment(phi[-1])
            if assigment is None:
                sync_failed = True
                break
            node_to_assignment[node] = assigment
        if sync_failed:
            continue
        sync_success += 1
        assigments = [node_to_assignment[node] for node in nodes]
        cut_size = calc_cut_size(connection_mat, assigments)
        if cut_size == max_cut_size:
            correct += 1
    print(f'Sync success rate = {sync_success / N_TIRAL * 100}%')
    print(f'Correct rate = {correct / N_TIRAL * 100}%')

if __name__ == '__main__':
    main()
