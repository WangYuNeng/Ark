from typing import Any

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.specification.cdg_types import EdgeType, NodeType


def create_connected_grid(
    n_rows: int,
    n_cols: int,
    node_type: NodeType,
    node_attrs: dict[str, Any],
    edge_type: EdgeType,
    edge_attrs: dict[str, Any],
    length: int,
    bidirectional_edge: bool = False,
) -> tuple[CDG, list[list[CDGNode]], list[CDGEdge]]:
    """Create a connected grid CDG with a given node, edge type, and side length.

    The grid is neighboring connected in a square with specified side length.
    The edge can be bidirectional, i.e., need only one edge to represent the src-to-dst and dst-to-src dynamics,
    e.g., OBC. Otherwise, the contribution from src to dst and dst to src is different and needs two edges, e.g.,
    CNN and NNL.

    Args:
        n_rows (int): # of rows
        n_cols (int): # of columns
        node_type (NodeType): grid node type
        node_attrs (dict[str, Any]): default node attributes
        edge_type (EdgeType): grid edge type
        edge_attrs (dict[str, Any]): default edge attributes
        length (int): side length of the grid, should be an even number.
        bidirectional_edge (bool, optional): whether the edge is bidirectional, i.e., need only
        one edge to represent the src-to-dst and dst-to-src dynamics (e.g., OBC). Defaults to False.
    Returns:
        tuple[CDG, list[list[CDGNode]], list[CDGEdge]]: the grid cdg, the grid nodes and edges
    """

    cdg = CDG()
    nodes = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    edges = []
    assert length % 2 == 0, "The length should be an even number."
    radius = length // 2

    def get_neighbors(row: int, col: int) -> list[tuple[int, int]]:
        neighbors = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0:
                    continue
                if 0 <= row + i < n_rows and 0 <= col + j < n_cols:
                    neighbors.append((row + i, col + j))
        return neighbors

    def new_edge():
        edges.append(edge_type(**edge_attrs))
        return edges[-1]

    for row in range(n_rows):
        for col in range(n_cols):
            nodes[row][col] = node_type(**node_attrs)
            self_edge = new_edge()
            cdg.connect(
                self_edge,
                nodes[row][col],
                nodes[row][col],
            )

    for row in range(n_rows):
        for col in range(n_cols):
            node = nodes[row][col]
            for n_row, n_col in get_neighbors(row, col):
                neighbor = nodes[n_row][n_col]
                edge = new_edge()
                cdg.connect(
                    edge,
                    node,
                    neighbor,
                )
                if bidirectional_edge:
                    edge = new_edge()
                    cdg.connect(
                        edge,
                        neighbor,
                        node,
                    )

    return cdg, nodes, edges

    return cdg, nodes, edges
