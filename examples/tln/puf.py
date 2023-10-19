from abc import ABC, abstractmethod
from functools import partial
from typing import Mapping, Optional

import numpy as np
import numpy.typing as npt
from spec import mm_tln_spec, pulse
from tqdm import tqdm

from ark.ark import Ark
from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.specification import CDGSpec


def bits2int(bits: list[bool], msb_first: bool = True) -> int:
    """Convert a base-2 representation in bit array to an integer

    Args:
        bits (list[bool]): base-2 representation, MSB first if msb_first is True
        msb_first (bool, optional): Whether the bits start with MSB. Defaults to True.

    Returns:
        int: The decimal integer value
    """
    n_bits = len(bits)
    if not msb_first:
        bit_enum = enumerate(bits[::-1])
    else:
        bit_enum = enumerate(bits)
    return sum([int(bit) * 2 ** (n_bits - i - 1) for i, bit in bit_enum])


def int2bits(val: int, n_bits: int, msb_first: bool = True) -> npt.NDArray[np.bool_]:
    """Convert an integer to base-2 representation

    Args:
        val (int): Decimal integer value
        n_bits (int): # of bits
        msb_first (bool, optional): whether the bit array starts with MSB.
        Defaults to True.

    Returns:
        npt.NDArray[np.bool_]: value in base-2 representation, MSB first if msb_first
        is True, LSB first otherwise
    """
    bits = []
    base = 2 ** (n_bits - 1)
    for i in range(n_bits):
        bits.append(int(val >= base))
        if val >= base:
            val -= base
        base //= 2
    if not msb_first:
        bits.reverse()
    return np.array(bits, dtype=np.bool_)


def single_bit_flipped_neighbors(chl: int, n_bits: int) -> list[int]:
    """Return the neighbors of the challenge with one bit flipped.

    Args:
        chl (int): The challenge.

    Returns:
        list[int]: The neighbors of the challenge.
    """
    neighbors = []
    for i in range(n_bits):
        neighbors.append(chl ^ (1 << i))
    return neighbors


def single_bit_flip_test(
    n_chl_bit: int, crps: Mapping[int, npt.NDArray[np.bool_]], center_chls: list[int]
) -> list[list[float]]:
    """Perform 1-bit flipping test from Uli's paper.

    Args:
        n_chl_bit (int): Number of challenge bits.
        crps (Mapping[int, npt.NDArray[np.bool_]]): Mapping from challenges to.
        response(s). The response can be multiple bits.
        center_chls (list[int]): Center challenge values.
    Returns:
        list[list[float]]: (n_chl_bit, n_rsp_bit) Flipping probability of each
        response bit for each challenge position.
    """

    n_rsp_bit = len(crps[center_chls[0]])
    flipped_cnt = np.zeros(shape=(n_chl_bit, n_rsp_bit))
    for chl in center_chls:
        chl_bits = int2bits(chl, n_chl_bit)
        rsp = crps[chl]
        for i, _ in enumerate(chl_bits):
            chl_bits[i] ^= 1
            flipped_chl = bits2int(chl_bits)
            rsp_flipped = crps[flipped_chl]
            flipped_cnt[i] += rsp != rsp_flipped
            chl_bits[i] ^= 1
    flipped_prob = flipped_cnt / len(center_chls)
    return flipped_prob


class PUF(ABC):
    def __init__(self, n_chl_bits: int, n_rsp_bits: int) -> None:
        self.n_chl_bits = n_chl_bits
        self.n_rsp_bits = n_rsp_bits

    @abstractmethod
    def sample_instances(self, n_inst: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def evaluate_instance(self, inst_id: int, challenge: int) -> list[bool]:
        pass


class SwitchableStarPUF(PUF):
    def __init__(
        self,
        n_chl_bits: int,
        n_rsp_bits: int,
        line_len: int,
        spec: CDGSpec,
        time_points: npt.NDArray[np.float64],
        do_validate: bool = False,
    ) -> None:
        """Initialize a nominal switchable star PUF.

        Args:
            n_chl_bits (int): # of challenge bits.
            n_rsp_bits (int): # of response bits.
            line_len (int): The length of the branch.
            spec (CDGSpec): The specification to define node types and edge types.
            time_points (npt.NDArray[np.float64]): The time points to evaluate when
            simulation
            do_validate (bool, optional): Whether to perform validation against the
            rules. Defaults to False.
        """
        super().__init__(n_chl_bits, n_rsp_bits)
        self.v_nt = spec.node_type("MmV")
        self.i_nt = spec.node_type("MmI")
        self.et = spec.edge_type("MmE")
        self.self_et = spec.edge_type("IdealE")
        self.inp_nt = spec.node_type("InpI")
        (
            self.puf_cdg,
            self.middle_caps,
            self.switch_pairs,
            self.branch_paris,
        ) = create_switchable_star_cdg(
            n_bits=n_chl_bits,
            line_len=line_len,
            i_nt=self.i_nt,
            v_nt=self.v_nt,
            et=self.et,
            self_et=self.self_et,
            inp_nt=self.inp_nt,
        )
        self.system = Ark(cdg_spec=spec)
        if do_validate:
            assert self.system.validate(self.puf_cdg)
        self.system.compile(self.puf_cdg)
        self.puf_cdg.initialize_all_states(val=0)
        self.time_points = time_points
        self.init_val = self.puf_cdg.execution_data()[0]
        self.line_len = line_len

    def sample_instances(self, n_inst: int, seed: int | None = None):
        """Sample n instances of the PUF from the nominal value and distribution.

        Args:
            n_inst (int): # of instances to sample
            seed (int | None, optional): Random seed. Defaults to None.
        """
        super().sample_instances(n_inst, seed)
        self.sampled_params = [self.puf_cdg.execution_data()[2] for _ in range(n_inst)]

    def evaluate_instance(self, inst_id: int, challenge: int) -> list[bool]:
        """Evaluate a PUF instance with a challenge.

        Args:
            inst_id (int): The id of the instance to evaluate.
            challenge (int): Challnege value.

        Returns:
            list[bool]: Response bits.
        """
        challenge_bits = int2bits(challenge, self.n_chl_bits)
        switch_val = {}
        for i, bit in enumerate(challenge_bits):
            switch_val[self.switch_pairs[0][i].name] = bit
            switch_val[self.switch_pairs[1][i].name] = bit
        exec_data = self.init_val, switch_val, self.sampled_params[inst_id]
        node_to_trace = self.system.execute(
            cdg_execution_data=exec_data,
            time_eval=self.time_points,
            store_inplace=False,
            max_step=2e-9,
        )
        cap_traces = [node_to_trace[cap.name] for cap in self.middle_caps]
        rsps = (np.cumsum(cap_traces[1]) - np.cumsum(cap_traces[0])) > 0
        return rsps

    def set_circuit_param(
        self,
        middle_cap_param: dict[str, float],
        middle_edge_param: dict[str, float],
        branch_v_param: list[dict[str, float]],
        branch_i_param: list[dict[str, float]],
        branch_e_param: list[dict[str, float]],
    ) -> None:
        """Set the nominal parameter value of the PUF.

        The total number of parameters equal the number of parameters on one branch.
        All branches have the same nominal parameters.

        Args:
            middle_cap_param (dict[str, float]): The 'c' and 'g' value of the middle cap.
            middle_edge_param (dict[str, float]): The 'ws' and 'wt' value of the
            middle edge.
            branch_v_param (list[dict[str, float]]): The 'c' and 'g' value of the
            capacitors on the branch.
            branch_i_param (list[dict[str, float]]): The 'l' and 'r' value of the
            inductors on the branch.
            branch_e_param (list[dict[str, float]]): The 'ws' and 'wt' value of the
            edges on the branch.
        """
        for cap in self.middle_caps:
            for attr, val in middle_cap_param.items():
                cap.set_attr_val(attr, val)
        for edges in self.switch_pairs:
            for edge in edges:
                for attr, val in middle_edge_param.items():
                    edge.set_attr_val(attr, val)
        for branches in self.branch_paris:
            for branch in branches:
                _, v_nodes, i_nodes, ets = branch
                for v_node, params in zip(v_nodes, branch_v_param):
                    for attr, val in params.items():
                        v_node.set_attr_val(attr, val)
                for i_node, params in zip(i_nodes, branch_i_param):
                    for attr, val in params.items():
                        i_node.set_attr_val(attr, val)
                for et, params in zip(ets, branch_e_param):
                    for attr, val in params.items():
                        et.set_attr_val(attr, val)

    @property
    def branch_n_nodes(self) -> int:
        return self.line_len

    @property
    def branch_n_edges(self) -> int:
        return 2 * self.line_len - 1


def create_branch(
    line_len: int, v_nt: NodeType, i_nt: EdgeType, et: EdgeType, self_et: EdgeType
) -> tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]:
    """Create a branch of a TLN

    Args:
        line_len (int): The length of the TLN.
        v_nt (NodeType): The node type of the capacitor nodes.
        i_nt (EdgeType): The edge type of the inductor nodes.
        et (EdgeType): The edge type of connections.
        self_et (EdgeType): The edge type of self-connections.

    Returns:
        tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]:
        The CDG, the capacitor nodes, the inductor nodes, and the edges.
    """
    # Create the CDG
    branch = CDG()
    # Create the nodes
    v_nodes = [v_nt() for _ in range(line_len)]
    i_nodes = [i_nt() for _ in range(line_len)]
    ets = [et() for _ in range(2 * line_len - 1)]
    # Create the edges
    for i in range(line_len):
        branch.connect(self_et(), v_nodes[i], v_nodes[i])
        branch.connect(self_et(), i_nodes[i], i_nodes[i])
        branch.connect(ets[2 * i], i_nodes[i], v_nodes[i])
        if not i == line_len - 1:
            branch.connect(ets[2 * i + 1], v_nodes[i], i_nodes[i + 1])
    return branch, v_nodes, i_nodes, ets


def create_switchable_star_cdg(
    n_bits,
    line_len,
    v_nt: NodeType,
    i_nt: EdgeType,
    et: EdgeType,
    self_et: EdgeType,
    inp_nt: NodeType,
) -> tuple[
    CDG,
    tuple[CDGNode, CDGNode],
    tuple[list[CDGEdge], list[CDGEdge]],
    tuple[
        list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
        list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
    ],
]:
    """Create a switchable star puf cdg with TLN

    Args:
        line_len (int): The length of the TLN.
        v_nt (NodeType): The node type of the capacitor nodes.
        i_nt (EdgeType): The edge type of the inductor nodes.
        et (EdgeType): The edge type of connections.
        self_et (EdgeType): The edge type of self-connections.
        inp_nt (NodeType): The node type of the input current node.

    Returns:
        CDG,
        tuple[CDGNode, CDGNode],
        tuple[list[CDGEdge], list[CDGEdge]],
        tuple[
            list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
            list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
        ]:
        The CDG, the middle capacitor nodes, the switch pairs correspond to bits, and
        all the branches' CDG, capacitor nodes, the inductor nodes, and edges.
    """
    branch_pairs = [
        [create_branch(line_len, v_nt, i_nt, et, self_et) for _ in range(n_bits)]
        for _ in range(2)
    ]

    puf = CDG()
    middle_caps = [v_nt() for _ in range(2)]
    switche_pairs = [[et(switchable=True) for _ in range(n_bits)] for _ in range(2)]
    short_pulse = partial(pulse, rise_time=0.5e-9, fall_time=0.5e-9, pulse_width=1e-9)
    for branches, cap, switches in zip(branch_pairs, middle_caps, switche_pairs):
        puf.connect(et(), inp_nt(fn=short_pulse, g=0.0), cap)
        puf.connect(et(), cap, cap)
        for branch, switch_edge in zip(branches, switches):
            branch_graph, _, i_nodes, _ = branch
            puf.add_graph(branch_graph)
            puf.connect(switch_edge, cap, i_nodes[0])
    return puf, middle_caps, switche_pairs, branch_pairs


if __name__ == "__main__":
    np.random.seed(428)
    n_bits = 12
    time_range = [0, 5e-8]
    time_points = np.linspace(*time_range, 1001, endpoint=True)
    ss_puf = SwitchableStarPUF(
        n_chl_bits=n_bits,
        n_rsp_bits=1,
        line_len=4,
        spec=mm_tln_spec,
        time_points=time_points,
    )
    vnode_param = {"c": 1e-9, "g": 0.0}
    inode_param = {"l": 1e-9, "r": 0.0}
    et_param = {"ws": 1.0, "wt": 1.0}
    middle_cap_param = vnode_param
    middle_edge_param = et_param
    branch_v_param = [vnode_param for _ in range(ss_puf.branch_n_nodes)]
    branch_i_param = [inode_param for _ in range(ss_puf.branch_n_nodes)]
    branch_e_param = [et_param for _ in range(ss_puf.branch_n_edges)]
    ss_puf.set_circuit_param(
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )
    ss_puf.sample_instances(n_inst=4)

    center_chls_size = 100
    center_chls = np.random.choice(
        2**n_bits, size=center_chls_size, replace=False
    ).tolist()
    neighbors = [single_bit_flipped_neighbors(chl, n_bits) for chl in center_chls]
    evaluate_chls = list(set(center_chls + sum(neighbors, [])))
    rsps_enumerate = [
        ss_puf.evaluate_instance(inst_id=0, challenge=i) for i in tqdm(evaluate_chls)
    ]
    crps = {
        chl: np.array(time_series_out)
        for chl, time_series_out in zip(evaluate_chls, rsps_enumerate)
    }
    flipped_prob = single_bit_flip_test(
        n_chl_bit=n_bits,
        crps=crps,
        center_chls=center_chls,
    )
    import matplotlib.pyplot as plt

    for bit_pos, prob in enumerate(flipped_prob):
        plt.plot(time_points, prob, label=f"Bit {bit_pos}")
    plt.show()
