import matplotlib.pyplot as plt
import numpy as np
from SALib import ProblemSpec
from spec import mm_tln_spec, pulse

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


def setup_prob(names, bounds, outputs):
    sp = ProblemSpec(
        {
            "names": names,
            "bounds": bounds,
            "outputs": outputs,
            "dists": ["norm" for _ in bounds],
        }
    )
    return sp


def plot_trace(save_path: str, sp: ProblemSpec, time_points, analyzed_times):
    name2s1trace, name2sttrace = (
        {name: [None for _ in analyzed_times] for name in analyzed_names},
        {name: [None for _ in analyzed_times] for name in analyzed_names},
    )

    for time_idx, time_name in enumerate(analyzed_times):
        s1s, sts = sp.analysis[time_name]["S1"], sp.analysis[time_name]["ST"]
        for name_idx, name in enumerate(analyzed_names):
            name2s1trace[name][time_idx] = s1s[name_idx]
            name2sttrace[name][time_idx] = sts[name_idx]

    for name, s1trace in name2s1trace.items():
        plt.plot(time_points, s1trace, label=name)
        plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("S1")
    plt.title("S1 vs time")
    plt.savefig(save_path + "_s1.png", dpi=300)
    plt.clf()
    # plt.show()
    for name, sttrace in name2sttrace.items():
        plt.plot(time_points, sttrace, label=name)
        plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("ST")
    plt.title("ST vs time")
    plt.savefig(save_path + "_st.png", dpi=300)
    plt.clf()
    # plt.show()


line_len = 4
time_range = [0, 2.5e-8]
time_points = np.linspace(*time_range, 101, endpoint=True)
analyzed_points = [i for i in range(len(time_points))]
analyzed_times = [f"{time_points[p]*1e9:.2f}ns" for p in analyzed_points]


# Single line setup


def create_single_line(line_len: int, tline: CDG):
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
    tline.initialize_all_states(0)
    return tline, source, v_nodes, i_nodes, edges


tline, source, v_nodes, i_nodes, edges = create_single_line(line_len, CDG())

system.compile(tline)
tline.initialize_all_states(0)


def tline_test_wl(l_w_arr, v_node_id, edge_id, integrate=False):
    traj_min_arr = []
    for row in l_w_arr:
        capacitance, ws, wt = row
        node2init, switch2val, element2attr = tline.execution_data()
        element2attr[v_nodes[v_node_id]]["c"] = capacitance
        element2attr[edges[edge_id]]["ws"] = ws
        element2attr[edges[edge_id]]["wt"] = wt
        exec_data = (node2init, switch2val, element2attr)
        node2trace = system.execute(
            cdg_execution_data=exec_data, time_eval=time_points, store_inplace=False
        )
        trace = node2trace[v_nodes[0]][0]
        trace = np.cumsum(trace)
        if integrate:
            trace = np.cumsum(trace)
        traj_min_arr.append([trace[p] for p in analyzed_points])
    return np.array(traj_min_arr)


sp = ProblemSpec(
    {
        "names": ["c[0]", "gm_s[1]", "gm_t[1]"],
        "bounds": [[1e-9, 0.1e-9], [1, 0.1], [1, 0.1]],
        "outputs": analyzed_times,
        "dists": ["norm" for _ in range(3)],
    }
)


def test_fn():
    pass


sp.sample_sobol(2**10).evaluate_parallel(test_fn, nprocs=4).analyze_sobol()


# Define the model inputs
for n_id, e_id in [(0, 1), (0, -1), (-1, 1), (-1, -1)]:
    print(f"Running for n_id: {n_id}, e_id: {e_id}")
    analyzed_names = [f"c[{n_id}]", f"gm_s[{e_id}]", f"gm_t[{e_id}]"]
    for integrate in [False, True]:
        sp = setup_prob(
            names=analyzed_names,
            bounds=[[1e-9, 0.1e-9], [1, 0.1], [1, 0.1]],
            outputs=analyzed_times,
        )
        sp.sample_sobol(2**10).evaluate_parallel(
            tline_test_wl, nprocs=4, v_node_id=n_id, edge_id=e_id, integrate=integrate
        ).analyze_sobol()
        if integrate:
            f_name = f"int_sobol_wl_nid_{n_id}_eid_{e_id}"
        else:
            f_name = f"sobol_wl_nid_{n_id}_eid_{e_id}"
        plot_trace(f_name, sp, time_points, analyzed_times)


def tline_test_ws(w_arr, integrate=False):
    traj_arr = []
    for row in w_arr:
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
        if integrate:
            trace = np.cumsum(trace)
        traj_arr.append([trace[p] for p in analyzed_points])
    return np.array(traj_arr)


n_vars = len(edges) * 2
analyzed_names = [f"gm_s[{i}]" for i in range(n_vars // 2)] + [
    f"gm_t[{i}]" for i in range(n_vars // 2)
]
for integrate in [False, True]:
    sp = setup_prob(
        names=analyzed_names,
        bounds=[[1, 0.1] for _ in range(n_vars)],
        outputs=analyzed_times,
    )
    (
        sp.sample_saltelli(2**10).evaluate_parallel(
            tline_test_ws, nprocs=4, integrate=integrate
        )
    ).analyze_sobol(nprocs=8)
    if integrate:
        f_name = "int_sobol_ws"
    else:
        f_name = "sobol_ws"
    plot_trace(f_name, sp, time_points, analyzed_times)


# Two line setup
line_len = 1


def create_two_line(line_len: int):
    tline, source, v_nodes, i_nodes, edges = create_single_line(line_len, CDG())
    tline, source2, v_nodes2, i_nodes2, edges2 = create_single_line(line_len, tline)
    sources = [source, source2]
    v_nodes = [v_nodes, v_nodes2]
    i_nodes = [i_nodes, i_nodes2]
    edges = [edges, edges2]
    return tline, sources, v_nodes, i_nodes, edges


tline, sources, v_nodes, i_nodes, edges = create_two_line(line_len)
system.compile(tline)
tline.initialize_all_states(0)


def test_lc_gm(lc_gm_arr, integrate=False):
    """Wrapping function for testing l, c, gm params of the two-lines setup"""
    traj_arr = []
    for row in lc_gm_arr:
        n_cs, n_ls, n_gms = len(v_nodes[0]), len(i_nodes[0]), len(edges[0])
        css, lss, gmss = (
            row[: n_cs * 2],
            row[n_cs * 2 : n_cs * 2 + n_ls * 2],
            row[n_cs * 2 + n_ls * 2 :],
        )
        node2init, switch2val, element2attr = tline.execution_data()
        for i, (vns_i, ins_i, es_i) in enumerate(zip(v_nodes, i_nodes, edges)):
            for j, node in enumerate(vns_i):
                element2attr[node]["c"] = css[i * n_cs + j]
            for j, node in enumerate(ins_i):
                element2attr[node]["l"] = lss[i * n_ls + j]
            for j, edge in enumerate(es_i):
                element2attr[edge]["ws"] = gmss[i * n_gms + j]
                element2attr[edge]["wt"] = gmss[i * n_gms + j + n_gms // 2]
        exec_data = (node2init, switch2val, element2attr)
        node2trace = system.execute(
            cdg_execution_data=exec_data, time_eval=time_points, store_inplace=False
        )
        trace = node2trace[v_nodes[0][0]][0] - node2trace[v_nodes[1][0]][0]
        if integrate:
            trace = np.cumsum(trace)
        traj_arr.append([trace[p] for p in analyzed_points])
    return np.array(traj_arr)


n_ci = len(v_nodes[0])
n_li = len(i_nodes[0])
n_gmi = len(edges[0]) * 2
n_vars = (n_ci + n_li + n_gmi) * 2
analyzed_names = (
    [f"c[{i // n_ci}][{i % n_ci}]" for i in range(2 * n_ci)]
    + [f"l[{i // n_li}][{i % n_li}]" for i in range(2 * n_li)]
    + [f"gm_s[0][{i}]" for i in range(n_gmi // 2)]
    + [f"gm_t[0][{i}]" for i in range(n_gmi // 2)]
    + [f"gm_s[1][{i}]" for i in range(n_gmi // 2)]
    + [f"gm_t[1][{i}]" for i in range(n_gmi // 2)]
)
bounds = (
    [[1e-9, 0.1e-9] for _ in range(2 * n_ci)]
    + [[1e-9, 0.1e-9] for _ in range(2 * n_li)]
    + [[1, 0.1] for _ in range(2 * n_gmi)]
)
for integrate in [False, True]:
    sp = setup_prob(
        names=analyzed_names,
        bounds=bounds,
        outputs=analyzed_times,
    )
    (
        sp.sample_saltelli(2**12).evaluate_parallel(
            test_lc_gm, nprocs=4, integrate=integrate
        )
    ).analyze_sobol(nprocs=8)
    if integrate:
        f_name = "int_sobol_all"
    else:
        f_name = "sobol_all"
    plot_trace(f_name, sp, time_points, analyzed_times)
