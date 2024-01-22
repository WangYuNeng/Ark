import os
import re

import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from spec import mm_tln_spec, pulse
from tqdm import tqdm

from ark.ark import Ark
from ark.cdg.cdg import CDG, CDGEdge, CDGNode

logger = Logging.setup_logging()


class Sampler:
    def __init__(self, seed) -> None:
        np.random.seed(seed=seed)

    def sample_float(self, low, high):
        return np.random.uniform(low=low, high=high)

    def sample_int(self, low, high):
        return np.random.randint(low=low, high=high)

    def sample_choice(self, choices, prob=None):
        return np.random.choice(a=choices, p=prob)


VNode, INode, Edge = (
    mm_tln_spec.node_type("MmV"),
    mm_tln_spec.node_type("MmI"),
    mm_tln_spec.edge_type("MmE"),
)
InpV = mm_tln_spec.node_type("InpV")


class Generator:
    def __init__(self) -> None:
        self._tln_spec = mm_tln_spec
        self._sampler = None

        self._type2elements = {}
        self._type2prefix = {
            VNode: "c",
            INode: "l",
            InpV: "s",
            Edge: "e",
        }

    def generate(self, max_op, seed) -> CDG:
        self._sampler = Sampler(seed=seed)
        graph = CDG()
        s = self._sampler
        self._type2elements = {VNode: [], INode: [], InpV: []}
        self._initialize(graph=graph)

        n_op = self._sampler.sample_int(0, max_op + 1)

        vnodes, inodes = self._type2elements[VNode], self._type2elements[INode]
        for _ in range(n_op):
            choices = vnodes + inodes
            node = s.sample_choice(choices=choices)
            # print(node)
            while not self._produce(graph, node):
                node = s.sample_choice(choices=choices)
        self._type2elements[Edge] = graph.edges
        return graph

    def _produce(self, graph: CDG, node: CDGNode):
        s = self._sampler

        def new_source(graph: CDG, node: CDGNode):
            src = self._gen_S(graph=graph)
            graph.connect(edge=Edge(wt=1.0, ws=1.0), src=src, dst=node)
            return True

        def new_vn_or_in(graph: CDG, node: CDGNode):
            if node.cdg_type == VNode:
                n_node = self._gen_IN(graph=graph)
            elif node.cdg_type == INode:
                n_node = self._gen_VN(graph=graph)
            else:
                assert False, "Only generate VN or IN node"
            return n_node

        def new_out(graph: CDG, node: CDGNode):
            n_node = new_vn_or_in(graph=graph, node=node)
            graph.connect(edge=Edge(wt=1.0, ws=1.0), src=node, dst=n_node)
            return True

        def new_in(graph: CDG, node: CDGNode):
            n_node = new_vn_or_in(graph=graph, node=node)
            graph.connect(edge=Edge(wt=1.0, ws=1.0), src=n_node, dst=node)
            return True

        def sample_valid_VN(node: CDGNode):
            if self._type2elements[VNode] == []:
                return False
            vnode = s.sample_choice(self._type2elements[VNode])
            if node.is_neighbor(vnode):
                return False
            return vnode

        def connect_out(graph: CDG, node: CDGNode):
            vnode = sample_valid_VN(node=node)
            if vnode:
                graph.connect(edge=Edge(wt=1.0, ws=1.0), src=node, dst=vnode)
                return True
            return False

        def connect_in(graph: CDG, node: CDGNode):
            vnode = sample_valid_VN(node=node)
            if vnode:
                graph.connect(edge=Edge(wt=1.0, ws=1.0), src=vnode, dst=node)
                return True
            return False

        def filter_choices(node: CDGNode, choices: set):
            this_type = node.cdg_type
            if node.cdg_type == VNode:
                choices.remove(connect_out)
                choices.remove(connect_in)

            for edge in node.edges:
                if edge.src == edge.dst:
                    continue
                if this_type == INode:
                    if node.is_src(edge=edge):
                        if new_out in choices:
                            choices.remove(new_out)
                        if connect_out in choices:
                            choices.remove(connect_out)
                    if node.is_dst(edge=edge):
                        if new_in in choices:
                            choices.remove(new_in)
                        if connect_in in choices:
                            choices.remove(connect_in)
                        if new_source in choices:
                            choices.remove(new_source)

            return choices

        choices = filter_choices(
            node=node,
            choices={new_source, new_out, new_in, connect_out, connect_in},
        )
        if len(choices) == 0:
            return False
        choice_func = s.sample_choice(choices=list(choices))
        return choice_func(graph, node)

    def _gen_S(self, graph: CDG):
        r = self._sampler.sample_float(0, 10)
        node = InpV(fn=pulse, r=r)
        self._type2elements[InpV].append(node)
        return node

    def _gen_VN(self, graph: CDG):
        c = self._sampler.sample_float(1e-10, 10e-9)
        g = self._sampler.sample_float(0, 10)
        node = VNode(c=c, g=g)
        graph.connect(edge=Edge(wt=1.0, ws=1.0), src=node, dst=node)
        self._type2elements[VNode].append(node)
        return node

    def _gen_IN(self, graph: CDG):
        l = self._sampler.sample_float(1e-10, 10e-9)
        r = self._sampler.sample_float(0, 10)
        node = INode(l=l, r=r)
        graph.connect(edge=Edge(wt=1.0, ws=1.0), src=node, dst=node)
        self._type2elements[INode].append(node)
        return node

    def _initialize(self, graph: CDG):
        s = self._sampler

        source = self._gen_S(graph=graph)
        sampled_type = s.sample_choice([INode, VNode], [0.5, 0.5])
        if sampled_type == INode:
            node = self._gen_IN(graph=graph)
        else:
            node = self._gen_VN(graph=graph)
        graph.connect(Edge(wt=1.0, ws=1.0), source, node)


class Simulator:
    def sim_cmp(self, spice_str: str, graph: CDG, data, tol: float):
        time_range = [0, 75e-9]

        system = Ark(cdg_spec=mm_tln_spec)
        system.validate(cdg=graph)
        system.compile(cdg=graph)

        circuit = Circuit("simulation")
        circuit.raw_spice += spice_str
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=1e-10, end_time=time_range[1])

        time_data = analysis.time.as_ndarray()
        traces = system.execute(
            cdg_execution_data=data, time_eval=time_data, store_inplace=False
        )

        all_ds_data, all_spice_data = [], []

        for node in graph.nodes_in_order(1):
            name = node.name.lower()
            ds_data = traces[node.name][0]
            spice_data = analysis.nodes[name].as_ndarray()
            all_ds_data.append(ds_data)
            all_spice_data.append(spice_data)

        all_ds_data = np.array(all_ds_data)
        all_spice_data = np.array(all_spice_data)

        error = np.mean(np.square((all_ds_data - all_spice_data))) / np.mean(
            np.square(all_spice_data)
        )
        if error > tol:
            plt.figure(1)
            for node in graph.nodes_in_order(1):
                ds_data = traces[node.name][0]
                spice_data = analysis.nodes[node.name.lower()].as_ndarray()
                plt.plot(time_data, ds_data, label="ds_{}".format(name))
                plt.plot(time_data, spice_data, label="spice_{}".format(name))
            plt.legend()
            plt.savefig("fail.png")
            plt.clf()
            assert False


GM_FACTOR = 1e-3
MODEL = mm_tln_spec

FUNC_PATTERN = r"([\w|_]+)=([0-9|e|\.|\+|\-]+)"
RE_OBJ = re.compile(FUNC_PATTERN)


class SpiceMapper:
    def __init__(self) -> None:
        pass

    def to_spice(self, graph: CDG, data):
        node: CDGNode
        spice_strs: list

        self._gmc_ins = set()
        self._gmrc_ins = set()

        self._graph = graph
        self._attr_val_map = data[2]
        spice_strs = []

        for node in self._graph.nodes:
            if node.cdg_type == VNode or node.cdg_type == INode:
                spice_strs.append(self._map_LC(node=node))

            elif node.cdg_type == InpV:
                spice_strs.append(self._map_src(node=node))

        for edge in self._graph.edges:
            pass
        spice_strs += self._base_ckt()

        return "\n".join(spice_strs)

    def _map_LC(self, node: CDGNode):
        attr_map = self._attr_val_map

        def calc_rloss():
            edge: CDGEdge

            if node.cdg_type == VNode:
                re_rloss = [attr_map[node.name]["g"]]
            elif node.cdg_type == INode:
                re_rloss = [attr_map[node.name]["r"]]
            for edge in node.edges:
                if edge.src.cdg_type == InpV:
                    if node.cdg_type == VNode:
                        inv_r = attr_map[edge.name]["wt"] / attr_map[edge.src.name]["r"]
                    elif node.cdg_type == INode:
                        inv_r = attr_map[edge.src.name]["r"] * attr_map[edge.name]["wt"]
                else:
                    continue

                re_rloss.append(inv_r)
            if re_rloss == []:
                return None
            return 1 / sum(re_rloss) / GM_FACTOR

        edge: CDGEdge
        component = "X{}".format(node.name)
        ins, gms = list(), list()
        for edge in node.edges:
            if edge.src == edge.dst:
                continue

            if node.is_src(edge=edge):
                in_name, in_gm = edge.dst.name, -attr_map[edge.name]["ws"] * GM_FACTOR
            else:
                in_name, in_gm = edge.src.name, attr_map[edge.name]["wt"] * GM_FACTOR

            ins.append(in_name)
            gms.append(in_gm)

        component = "X{}".format(node.name)

        if node.cdg_type == VNode:
            base_val = float(attr_map[node.name]["c"])
        elif node.cdg_type == INode:
            base_val = float(attr_map[node.name]["l"])

        rloss = calc_rloss()
        model_prefix = "gmrc"
        self._gmrc_ins.add(len(ins))
        params = "Cint={} Rloss={} ".format(base_val * GM_FACTOR, rloss) + " ".join(
            ["gm{}={}".format(i, gm) for i, gm in enumerate(gms)]
        )

        inputs = " ".join(ins)
        output = node.name
        model = "{}{}".format(model_prefix, len(ins))
        spice_str = " ".join([component, inputs, output, model, params])
        return spice_str

    def _map_src(self, node: CDGNode):
        def parse_attr(attr_str):
            matches = RE_OBJ.findall(attr_str)
            params = {m[0]: float(m[1]) for m in matches}
            return (
                params["amplitude"],
                params["delay"],
                params["rise_time"],
                params["fall_time"],
                params["pulse_width"],
                params["period"],
            )

        amplitude, delay, rise_time, fall_time, pulse_width, period = (
            1,
            0,
            5e-9,
            5e-9,
            10e-9,
            1,
        )
        assert len(node.edges) == 1
        for edge in node.edges:
            pass
        if edge.dst.cdg_type == VNode:
            amplitude /= float(self._attr_val_map[node.name]["r"])
        return "V{} {} 0 DC 0V PULSE(0V {}V {}s {}s {}s {}s {}s)".format(
            node.name,
            node.name,
            amplitude,
            delay,
            rise_time,
            fall_time,
            pulse_width,
            period,
        )

    def _base_ckt(self):
        sub_strs = []
        for n_in in self._gmrc_ins:
            vi_str = " ".join(["vi{}".format(i) for i in range(n_in)])
            gm_str = " ".join(["gm{}=1e-3".format(i) for i in range(n_in)])
            sub_strs.append(
                ".subckt gmrc{} {} vo Cint=1e-12 {} Rloss=1e12".format(
                    n_in, vi_str, gm_str
                )
            )
            sub_strs += [
                "Gvccs{} 0 vo vi{} 0 gm{}".format(i, i, i) for i in range(n_in)
            ]
            sub_strs.append("Rr vo 0 r=Rloss")
            sub_strs.append("Cc vo 0 c=Cint")
            sub_strs.append(".ends gmrc{}".format(n_in))

        return sub_strs


if __name__ == "__main__":
    g = Generator()
    sim = Simulator()

    mapper = SpiceMapper()

    if not os.path.exists("../../output/spice-dump"):
        os.mkdir("../../output/spice-dump")

    for seed in tqdm(range(1000)):
        graph = g.generate(max_op=20, seed=seed)
        graph.initialize_all_states(val=0)
        execution_data = graph.execution_data()
        spice_str = mapper.to_spice(graph=graph, data=execution_data)
        with open("../../output/spice-dump/tln-rand-inst{}.sp".format(seed), "w") as f:
            f.write(spice_str)

        sim.sim_cmp(spice_str=spice_str, graph=graph, data=execution_data, tol=0.01)
