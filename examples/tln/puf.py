from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from spec import mm_tln_spec, pulse
from tqdm import tqdm

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.trainable import Trainable, TrainableMgr


@dataclass
class PUFParams:

    mgr: TrainableMgr
    middle_cap: Trainable | float
    branch_caps: list[Trainable | float]
    branch_inds: list[Trainable | float]
    branch_gms: tuple[list[Trainable | float], list[Trainable | float]]


def create_branch(
    line_len: int,
    v_nt: NodeType,
    i_nt: EdgeType,
    et: EdgeType,
    self_et: EdgeType,
    puf_params: PUFParams,
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
    v_nodes = [v_nt(c=puf_params.branch_caps[i], g=0.0) for i in range(line_len)]
    i_nodes = [i_nt(l=puf_params.branch_inds[i], r=0.0) for i in range(line_len)]

    # Create the edges -- connece V-L-V-L-...,
    # Start with 1 to skip the edge connected the center (will generate
    # in the `create_switchable_star_cdg` function)
    ets = [
        et(ws=puf_params.branch_gms[0][i], wt=puf_params.branch_gms[1][i])
        for i in range(1, 2 * line_len)
    ]
    # Connect the edges
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
    init_caps: Optional[list[float]] = None,
    init_inds: Optional[list[float]] = None,
    init_gms: Optional[tuple[list[float], list[float]]] = None,
    pulse_params: tuple[float, float, float] = (0.5e-9, 0.5e-9, 1e-9),
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
        line_len (int): The length of the TLN.
        v_nt (NodeType): The node type of the capacitor nodes.
        i_nt (EdgeType): The edge type of the inductor nodes.
        et (EdgeType): The edge type of connections.
        self_et (EdgeType): The edge type of self-connections.
        inp_nt (NodeType): The node type of the input current node.
        init_caps (Optional[list[float]]): The initial values of the capacitors.
        init_inds (Optional[list[float]]): The initial values of the inductors.
        init_gms (Optional[tuple[list[float], list[float]]): The initial values of the
            transconductances.

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

    assert not init_caps or len(init_caps) == line_len + 1
    assert not init_inds or len(init_inds) == line_len
    assert not init_gms or (
        len(init_gms[0]) == 2 * line_len and len(init_gms[1]) == 2 * line_len
    )

    if not init_caps:
        init_caps = [None] * (line_len + 1)
    if not init_inds:
        init_inds = [None] * line_len
    if not init_gms:
        init_gms = ([None] * 2 * line_len, [None] * 2 * line_len)

    # Initialize all the trainable elements
    weight_mgr = TrainableMgr()
    puf_params = PUFParams(
        mgr=weight_mgr,
        middle_cap=init_caps[0] if init_caps[0] else weight_mgr.new_analog(),
        branch_caps=[
            init_caps[i] if init_caps[i] else weight_mgr.new_analog()
            for i in range(1, line_len + 1)
        ],
        branch_inds=[
            init_inds[i] if init_inds[i] else weight_mgr.new_analog()
            for i in range(line_len)
        ],
        branch_gms=(
            [
                init_gms[0][i] if init_gms[0][i] else weight_mgr.new_analog()
                for i in range(2 * line_len)
            ],
            [
                init_gms[1][i] if init_gms[1][i] else weight_mgr.new_analog()
                for i in range(2 * line_len)
            ],
        ),
    )

    # Create 2 * n_bits nominally identical branches
    branch_pairs = [
        [
            create_branch(line_len, v_nt, i_nt, et, self_et, puf_params)
            for _ in range(n_bits)
        ]
        for _ in range(2)
    ]

    puf = CDG()
    middle_caps = [v_nt(c=puf_params.middle_cap, g=0) for _ in range(2)]
    switche_pairs = [
        [
            et(
                switchable=True,
                ws=puf_params.branch_gms[0][0],
                wt=puf_params.branch_gms[1][0],
            )
            for _ in range(n_bits)
        ]
        for _ in range(2)
    ]
    rise_time, fall_time, pulse_width = pulse_params
    short_pulse = partial(
        pulse, rise_time=rise_time, fall_time=fall_time, pulse_width=pulse_width
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
