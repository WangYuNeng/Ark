from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import numpy as np
from spec import mm_tln_spec, pulse, unity
from tqdm import tqdm

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.trainable import Trainable, TrainableMgr


@dataclass
class PUFParams:

    mgr: TrainableMgr
    middle_cap: Trainable | float
    middle_g: Trainable | float
    branch_caps: list[Trainable | float]
    branch_gs: list[Trainable | float]
    branch_inds: list[Trainable | float]
    branch_rs: list[Trainable | float]
    branch_gms: tuple[list[Trainable | float], list[Trainable | float]]


def create_branch(
    line_len: int | float,
    v_nt: NodeType,
    i_nt: EdgeType,
    et: EdgeType,
    self_et: EdgeType,
    puf_params: PUFParams,
    gm_lut: Callable,
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
    n_cap, n_ind = int(np.floor(line_len)), int(np.ceil(line_len))
    n_gm = int(2 * line_len)
    # Create the nodes
    v_nodes = [
        v_nt(c=puf_params.branch_caps[i], g=puf_params.branch_gs[i])
        for i in range(n_cap)
    ]
    i_nodes = [
        i_nt(l=puf_params.branch_inds[i], r=puf_params.branch_rs[i])
        for i in range(n_ind)
    ]

    # Create the edges -- connece V-L-V-L-...,
    # Start with 1 to skip the edge connected the center (will generate
    # in the `create_switchable_star_cdg` function)
    ets = [
        et(
            ws=puf_params.branch_gms[0][i],
            wt=puf_params.branch_gms[1][i],
            gm_lut=gm_lut,
        )
        for i in range(1, n_gm)
    ]
    # Connect the edges
    for i in range(n_gm):
        node_i, is_v = divmod(i, 2)
        if is_v:
            branch.connect(self_et(), v_nodes[node_i], v_nodes[node_i])
            if i == n_gm - 1:
                break
            branch.connect(ets[i], v_nodes[node_i], i_nodes[node_i + 1])

        else:
            branch.connect(self_et(), i_nodes[node_i], i_nodes[node_i])
            if i == n_gm - 1:
                break
            branch.connect(ets[i], i_nodes[node_i], v_nodes[node_i])
    return branch, v_nodes, i_nodes, ets


def create_switchable_star_cdg(
    n_bits,
    line_len,
    v_nt: NodeType,
    i_nt: EdgeType,
    et: EdgeType,
    self_et: EdgeType,
    inp_nt: NodeType,
    fixed_caps: Optional[list[float] | float] = None,
    fixed_gs: Optional[list[float] | float] = None,
    fixed_inds: Optional[list[float] | float] = None,
    fixed_rs: Optional[list[float] | float] = None,
    fixed_gms: Optional[tuple[list[float], list[float]] | float] = None,
    pulse_params: tuple[float, float, float] = (0.5e-9, 0.5e-9, 1e-9),
    gm_lut: Optional[Callable] = unity,
) -> tuple[
    CDG,
    tuple[CDGNode, CDGNode],
    tuple[list[CDGEdge], list[CDGEdge]],
    tuple[
        list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
        list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
    ],
    PUFParams,
]:
    """Create a switchable star puf cdg with TLN

    Args:
        line_len (int): The length of the TLN. If the length is x.5, the TLN will have
            floor(x) capacitors and ceil(x) inductors on a branch (excluding the center)
        v_nt (NodeType): The node type of the capacitor nodes.
        i_nt (EdgeType): The edge type of the inductor nodes.
        et (EdgeType): The edge type of connections.
        self_et (EdgeType): The edge type of self-connections.
        inp_nt (NodeType): The node type of the input current node.
        fixed_caps (Optional[list[float]] | float): The values of capacitors that are fixed during training.
            if the value is flo
        fixed_gs (Optional[list[float]]) | float: The values of conductances associated to capacitors that are
            fixed during training.
        fixed_inds (Optional[list[float]] | float): The values of inductors that are fixed during training.
        fixed_rs (Optional[list[float]] | float): The values of resistances associated to inductors that are
            fixed during training.
        fixed_gms (Optional[tuple[list[float], list[float]] | float): The values of the
            transconductances that are fixed during training.
        pulse_params (tuple[float, float, float]): The pulse parameters for the input
            waveform.
        gm_lut (Optional[Callable]): The lookup table for the transconductance
            calculation.

    Returns:
        CDG,
        tuple[CDGNode, CDGNode],
        tuple[list[CDGEdge], list[CDGEdge]],
        tuple[
            list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
            list[tuple[CDG, list[CDGNode], list[CDGNode], list[CDGEdge]]],
        ],
        PUFParams: The CDG, the middle capacitor nodes, the switch pairs correspond to bits,
        and all the branches' CDG, capacitor nodes, the inductor nodes, the edges, and the
        symbolic PUF parameters.
    """

    # line_len must be an integer or x.5
    assert int(line_len) == line_len or line_len * 2 == int(line_len) * 2 + 1
    line_n_cap = int(np.floor(line_len)) + 1
    line_n_ind = int(np.ceil(line_len))
    line_n_gm = int(2 * line_len)

    if not isinstance(fixed_caps, list):
        fixed_caps = [fixed_caps] * line_n_cap
    if not isinstance(fixed_inds, list):
        fixed_inds = [fixed_inds] * line_n_ind
    if not isinstance(fixed_gms, tuple):
        fixed_gms = ([fixed_gms] * line_n_gm, [fixed_gms] * line_n_gm)
    if not isinstance(fixed_gs, list):
        fixed_gs = [fixed_gs] * line_n_cap
    if not isinstance(fixed_rs, list):
        fixed_rs = [fixed_rs] * line_n_ind

    # Initialize all the trainable elements
    weight_mgr = TrainableMgr()
    puf_params = PUFParams(
        mgr=weight_mgr,
        middle_cap=fixed_caps[0] if fixed_caps[0] else weight_mgr.new_analog(),
        middle_g=fixed_gs[0] if fixed_gs[0] else weight_mgr.new_analog(),
        branch_caps=[
            fixed_caps[i] if fixed_caps[i] else weight_mgr.new_analog()
            for i in range(1, line_n_cap)
        ],
        branch_gs=[
            fixed_gs[i] if fixed_gs[i] else weight_mgr.new_analog()
            for i in range(1, line_n_cap)
        ],
        branch_inds=[
            fixed_inds[i] if fixed_inds[i] else weight_mgr.new_analog()
            for i in range(line_n_ind)
        ],
        branch_rs=[
            fixed_rs[i] if fixed_rs[i] else weight_mgr.new_analog()
            for i in range(line_n_ind)
        ],
        branch_gms=(
            [
                fixed_gms[0][i] if fixed_gms[0][i] else weight_mgr.new_analog()
                for i in range(line_n_gm)
            ],
            [
                fixed_gms[1][i] if fixed_gms[1][i] else weight_mgr.new_analog()
                for i in range(line_n_gm)
            ],
        ),
    )

    # Create 2 * n_bits nominally identical branches
    branch_pairs = [
        [
            create_branch(line_len, v_nt, i_nt, et, self_et, puf_params, gm_lut)
            for _ in range(n_bits)
        ]
        for _ in range(2)
    ]

    puf = CDG()
    middle_caps = [
        v_nt(c=puf_params.middle_cap, g=puf_params.middle_g) for _ in range(2)
    ]
    switche_pairs = [
        [
            et(
                switchable=True,
                ws=puf_params.branch_gms[0][0],
                wt=puf_params.branch_gms[1][0],
                gm_lut=gm_lut,
            )
            for _ in range(n_bits)
        ]
        for _ in range(2)
    ]
    amplitude, rise_time, fall_time, pulse_width = pulse_params
    short_pulse = partial(
        pulse,
        amplitude=amplitude,
        rise_time=rise_time,
        fall_time=fall_time,
        pulse_width=pulse_width,
    )
    for branches, cap, switches in zip(branch_pairs, middle_caps, switche_pairs):
        # Assume the input current input is ideal for simplicity
        puf.connect(self_et(), inp_nt(fn=short_pulse, g=0.0), cap)
        puf.connect(self_et(), cap, cap)
        for branch, switch_edge in zip(branches, switches):
            branch_graph, _, i_nodes, _ = branch
            puf.add_graph(branch_graph)
            puf.connect(switch_edge, cap, i_nodes[0])
    return puf, middle_caps, switche_pairs, branch_pairs, puf_params
