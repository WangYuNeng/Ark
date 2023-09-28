"""
Example: Coupled Oscillator Network for solving the max-cut problem
ref: https://ieeexplore.ieee.org/document/9531734
"""

from argparse import ArgumentParser
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from spec import T, calc_offset_std, coupling_fn, hw_obc_spec, locking_fn
from tqdm import tqdm

from ark.ark import Ark
from ark.cdg.cdg import CDG
from ark.specification.cdg_types import EdgeType, NodeType

obc_spec = hw_obc_spec
help_fn = [locking_fn, coupling_fn]
Osc, Coupling = obc_spec.node_type("Osc"), obc_spec.edge_type("Coupling")
Coupling_offset = obc_spec.edge_type("Coupling_offset")

parser = ArgumentParser()
parser.add_argument("--baseline", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-n", "--n_trial", type=int, default=1000)
parser.add_argument(
    "--n_cycle", type=int, default=5, help="Number of cycles to run the simulation"
)
parser.add_argument("--atol", type=float, default=1e-2)
parser.add_argument("--rtol", type=float, default=1e-2)
parser.add_argument(
    "--offset_rstd",
    type=float,
    default=None,
    help="Standard deviation of the offset relative to the sine function. \
                          The disctibution is N(0, offset_rstd * 2 * F_2) \
                          The 2 * F_2 term is to make the mismatch same order as scale_rstd.",
)
parser.add_argument("--initialize", type=float, default=None)
parser.add_argument(
    "--w_n_bits", type=int, default=1, help="Number of bits for cut weights"
)
sim_args = parser.parse_args()

BASELINE = sim_args.baseline
N_CYCLE = sim_args.n_cycle
PLOT = sim_args.plot
N_TIRAL = sim_args.n_trial
ATOL, RTOL = sim_args.atol, sim_args.rtol
OFFSET_RSTD = sim_args.offset_rstd
INITIALIZE = sim_args.initialize
W_N_BITS = sim_args.w_n_bits

# Change the offset std value if specified
if OFFSET_RSTD is not None:
    OFFSET_STD = calc_offset_std(OFFSET_RSTD)
    Coupling_offset.attr_def["offset"].std = OFFSET_STD


def create_max_cut_con(connection_mat, osc_nt: NodeType, cp_et: EdgeType):
    """Create a CDG of con for solving MAXCUT of the graph described by connection_mat"""
    nodes = [osc_nt(lock_fn=locking_fn, osc_fn=coupling_fn) for _ in range(4)]
    if cp_et == Coupling:
        args = {"k": -1.0}
    elif cp_et == Coupling_offset:
        args = {"k": -1.0, "offset": 0.0}

    graph = CDG()
    for i, row in enumerate(connection_mat):
        for j, val in enumerate(row):
            if val:
                args["k"] = float(-val)
                graph.connect(cp_et(**args), nodes[i], nodes[j])
    for i in range(4):
        graph.connect(Coupling(k=1.0), nodes[i], nodes[i])

    return nodes, graph


def calc_cut_size(connection_mat, cut):
    """Calculate the cut size of a cut"""
    cut_val = 0
    for i, ass0 in enumerate(cut):
        for j, ass1 in enumerate(cut[i + 1 :]):
            # If the nodes are in different subsets, add the edge value to the cut value
            if ass0 != ass1:
                cut_val += connection_mat[i][i + 1 + j]
    return cut_val


def gen_max_cut_prob(seed, w_n_bits=W_N_BITS):
    """Generate a max cut problem"""
    np.random.seed(seed)
    connection_mat = np.random.randint(0, 2**w_n_bits, size=(4, 4))
    # set diagonal and lower triangle to 0
    connection_mat = np.triu(connection_mat, 1)

    cut_enumeration = [[a, b, c, 0] for a, b, c in product([0, 1], repeat=3)]
    max_cut, max_cut_size = [0 for _ in range(4)], 0
    for cut in cut_enumeration:
        cut_val = calc_cut_size(connection_mat, cut)
        if cut_val > max_cut_size:
            max_cut, max_cut_size = cut, cut_val
    return connection_mat, max_cut_size, max_cut


def plot_oscillation(time_points, cdg: CDG, omega, scaling, title=None):
    """Plot the oscillation of the oscillator"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update(
        {"text.usetex": True, "font.size": 15, "font.family": "Helvetica"}
    )
    cycle = time_points / T
    fig, ax = plt.subplots(nrows=2)
    for node in cdg.stateful_nodes():
        phi = node.get_trace(n=0) * scaling
        ax[1].plot(cycle, np.sin(omega * time_points + phi), label=node.name)
        ax[0].plot(cycle, phi)
    ax[0].set_title(r"$\phi$")
    ax[0].set_ylabel("phase (rad)")
    ax[1].set_title(r"$\sin{(\omega t + \phi)}$")
    ax[1].set_ylabel("Amplitude (V)")
    ax[1].set_xlabel("\# of cycle (t/T)")
    plt.tight_layout()
    if title:
        plt.savefig(title + ".pdf")
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
    system = Ark(cdg_spec=obc_spec)

    if BASELINE:
        correct = 0
        for prob in tqdm(problems, desc="Baseline (random guess)", total=N_TIRAL):
            connection_mat, max_cut = prob
            assignments = np.random.randint(0, 2, size=4)
            cut_size = calc_cut_size(connection_mat, assignments)
            if cut_size == max_cut:
                correct += 1
        print(f"Correct rate: {correct / N_TIRAL * 100}%")
        exit()

    if OFFSET_RSTD:
        cp_et = Coupling_offset
    else:
        cp_et = Coupling

    osc_nodetype = Osc
    cycle = T
    omega = 2 * np.pi / T
    scaling = np.pi

    correct = 0
    sync_success = 0
    for seed, prob in tqdm(
        enumerate(problems), desc=f"{osc_nodetype.name}, {cp_et.name}", total=N_TIRAL
    ):
        connection_mat, max_cut_size, max_cut = prob
        nodes, graph = create_max_cut_con(connection_mat, osc_nodetype, cp_et)

        system.compile(cdg=graph)
        time_range = [0, N_CYCLE * cycle]
        time_points = np.linspace(*time_range, 1000)
        if INITIALIZE is not None:
            graph.initialize_all_states(val=INITIALIZE)
        else:
            np.random.seed(seed)
            graph.initialize_all_states(rand=True)
        system.execute(cdg=graph, time_eval=time_points, init_seed=seed)
        if seed == 1 and PLOT:
            plot_oscillation(
                time_points,
                graph,
                omega,
                scaling,
                title=f"{osc_nodetype.name}, {cp_et.name}",
            )
        node_to_assignment = {}
        sync_failed = False
        for node in graph.stateful_nodes():
            phi = node.get_trace(n=0) * scaling
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
    print(f"Sync success rate = {sync_success / N_TIRAL * 100}%")
    print(f"Correct rate = {correct / N_TIRAL * 100}%")


if __name__ == "__main__":
    main()
