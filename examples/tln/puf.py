from spec import mm_tln_spec, pulse

from ark.cdg.cdg import CDG, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType


def create_branch(
    line_len: int, v_nt: NodeType, i_nt: EdgeType, et: EdgeType
) -> tuple[CDG, NodeType]:
    """Create a branch of a TLN

    Args:
        line_len (int): The length of the TLN
        v_nt (NodeType): The node type of the capacitor nodes
        i_nt (EdgeType): The edge type of the inductor nodes
        et (EdgeType): The edge type of connections

    Returns:
        CDG: The CDG of the branch
        first_i_node: The first inductor node of the branch (for connecting to the
        other branches)
    """
    # Create the CDG
    branch = CDG()
    # Create the nodes
    v_nodes = [v_nt() for _ in range(line_len)]
    i_nodes = [i_nt() for _ in range(line_len)]
    # Create the edges
    for i in range(line_len):
        branch.connect(et(), v_nodes[i], v_nodes[i])
        branch.connect(et(), i_nodes[i], i_nodes[i])
        branch.connect(et(), i_nodes[i], v_nodes[i])
        if not i == line_len - 1:
            branch.connect(et(), v_nodes[i], i_nodes[i + 1])
    return branch, i_nodes[0]


def create_switchable_star_cdg(
    n_bits, line_len, v_nt: NodeType, i_nt: EdgeType, et: EdgeType, inp_nt: NodeType
) -> tuple[CDG, tuple[CDGNode, CDGNode]]:
    two_lines = [
        [create_branch(line_len, v_nt, i_nt, et) for _ in range(n_bits)]
        for _ in range(2)
    ]

    puf = CDG()
    middle_caps = [v_nt() for _ in range(2)]
    for lines, cap in zip(two_lines, middle_caps):
        puf.connect(et(), inp_nt(fn=pulse, g=0.0), cap)
        puf.connect(et(), cap, cap)
        for line_node in lines:
            line, i_node = line_node
            puf.add_graph(line)
            puf.connect(et(), cap, i_node)
    return puf, middle_caps


if __name__ == "__main__":
    v_nt = mm_tln_spec.node_type("IdealV")
    i_nt = mm_tln_spec.node_type("IdealI")
    et = mm_tln_spec.edge_type("IdealE")
    inp_nt = mm_tln_spec.node_type("InpI")

    puf, middle_caps = create_switchable_star_cdg(
        n_bits=1, line_len=10, v_nt=v_nt, i_nt=i_nt, et=et, inp_nt=inp_nt
    )
    import numpy as np

    from ark.ark import Ark

    for node in puf.nodes:
        if node.cdg_type == v_nt:
            node.set_attr_val("c", 1e-9)
            node.set_attr_val("g", 0.0)
        elif node.cdg_type == i_nt:
            node.set_attr_val("l", 1e-9)
            node.set_attr_val("r", 0.0)
    # for edge in puf.edges:
    #     edge.set_attr_val("ws", 1.0)
    #     edge.set_attr_val("wt", 1.0)

    system = Ark(cdg_spec=mm_tln_spec)
    assert system.validate(puf)
    system.compile(puf, verbose=True)
    puf.initialize_all_states(val=0)
    time_range = [0, 5e-8]
    time_points = np.linspace(*time_range, 201, endpoint=True)
    node2trace = system.execute(
        cdg_execution_data=puf.execution_data(),
        time_eval=time_points,
        store_inplace=False,
        max_step=1e-10,
    )
    import matplotlib.pyplot as plt

    plt.plot(time_points, node2trace[middle_caps[0].name][0])
    plt.plot(time_points, node2trace[middle_caps[1].name][0])
    plt.show()
