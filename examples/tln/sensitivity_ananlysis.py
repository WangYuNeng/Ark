import time

import matplotlib.pyplot as plt
import numpy as np
from SALib import ProblemSpec
from spec import mm_tln_spec, pulse
from tqdm import tqdm

from ark.ark import Ark
from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.specification.attribute_def import AttrDef
from ark.specification.range import Range

spec = mm_tln_spec
IdealV, IdealI = spec.node_type("IdealV"), spec.node_type("IdealI")
InpI = spec.node_type("InpI")
MmE = spec.edge_type("MmE")

MmE.attr_def = {
    "ws": AttrDef("ws", attr_type=float, attr_range=Range(min=0.5, max=1.5)),
    "wt": AttrDef("wt", attr_type=float, attr_range=Range(min=0.5, max=1.5)),
}


system = Ark(cdg_spec=spec)


line_len = 2

tline = CDG()
source = InpI(fn=pulse, g=0.0)
edges: list[CDGEdge] = [MmE(ws=1.0, wt=1.0)]
v_nodes: list[CDGNode] = [IdealV(c=1e-9, g=0.0)]
i_nodes: list[CDGNode] = []

tline.connect(edges[0], source, v_nodes[0])
for i in range(line_len):
    v_nodes.append(IdealV(c=1e-9, g=0.0))
    i_nodes.append(IdealI(l=1e-9, r=0.0))
    edges.extend([MmE(ws=1.0, wt=1.0), MmE(ws=1.0, wt=1.0)])
    tline.connect(edges[-2], v_nodes[-2], i_nodes[-1])
    tline.connect(edges[-1], i_nodes[-1], v_nodes[-1])

system.compile(tline)
tline.initialize_all_states(0)
time_range = [0, 1e-8]
time_points = np.linspace(*time_range, 100)


def tline_test_wl(l_w_arr):
    traj_min_arr = []
    for row in tqdm(l_w_arr):
        inductance, ws, wt = row
        node2init, switch2val, element2attr = tline.execution_data()
        element2attr[i_nodes[-1]]["l"] = inductance
        element2attr[edges[-1]]["ws"] = ws
        element2attr[edges[-1]]["wt"] = wt
        exec_data = (node2init, switch2val, element2attr)
        node2trace = system.execute(
            cdg_execution_data=exec_data, time_eval=time_points, store_inplace=False
        )
        trace = node2trace[v_nodes[0]][0]
        # traj_min_arr.append([max(trace), trace[25], trace[50], trace[75], trace[-1]])
        traj_min_arr.append(trace[75])
    return np.array(traj_min_arr)


def tline_test_ws(w_arr):
    traj_min_arr = []
    for row in tqdm(w_arr):
        n_weights = len(row)
        assert n_weights == len(edges) * 2
        wss, wts = row[: n_weights // 2], row[n_weights // 2 :]
        assert len(wss) == len(wts) == len(edges)
        node2init, switch2val, element2attr = tline.execution_data()
        for i, edge in enumerate(edges):
            element2attr[edge]["ws"] = wss[i]
            element2attr[edge]["wt"] = wts[i]
        exec_data = (node2init, switch2val, element2attr)
        node2trace = system.execute(
            cdg_execution_data=exec_data, time_eval=time_points, store_inplace=False
        )
        trace = node2trace[v_nodes[0]][0]
        traj_min_arr.append([max(trace), trace[25], trace[50], trace[75], trace[-1]])
    return np.array(traj_min_arr)


# Define the model inputs
sp = ProblemSpec(
    {
        "names": ["l", "ws", "wt"],
        "bounds": [[0.8e-9, 1.2e-9], [0.8, 1.2], [0.8, 1.2]],
        # "outputs": ["max", "2.5ns", "5ns", "7.5ns", "10ns"],
    }
)

sp.sample_sobol(2**10).evaluate_parallel(tline_test_wl, nprocs=4).analyze_rsa(bins=2)
print(sp)
sp.plot()
plt.show()


n_vars = 2 * len(edges)
sp = ProblemSpec(
    {
        "names": [f"ws_{i}" for i in range(n_vars // 2)]
        + [f"wt_{i}" for i in range(n_vars // 2)],
        "bounds": [[0.8, 1.2] for _ in range(n_vars)],
        "outputs": ["max", "2.5ns", "5ns", "7.5ns", "10ns"],
    }
)
start_time = time.perf_counter()
(sp.sample_saltelli(1024).evaluate(tline_test_ws)).analyze_sobol()
print(f"Time taken with 1 core: {time.perf_counter() - start_time:.2f}s")

start_time = time.perf_counter()
(sp.sample_saltelli(1024).evaluate_parallel(tline_test_ws, nprocs=4)).analyze_sobol()
print(f"Time taken with 4 core: {time.perf_counter() - start_time:.2f}s")
print(sp)
sp.plot()
plt.show()
