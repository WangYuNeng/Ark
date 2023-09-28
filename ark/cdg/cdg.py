from typing import Mapping

import numpy as np

from ark.reduction import Reduction
from ark.specification.attribute_def import AttrDef, AttrImpl
from ark.specification.rule_keyword import DST, SELF, SRC, Target


class CDGElement:
    """Base class for CDG nodes and edges."""

    attr_def: Mapping[str, AttrDef]

    def __init__(
        self, cdg_type: "CDGType", name: str, **attrs: Mapping[str, AttrImpl]
    ) -> None:
        self.cdg_type = cdg_type
        self.name = name
        self.attrs = attrs

    def __str__(self) -> str:
        return self.name

    def get_attr_str(self, attr_name: str) -> str:
        """Return the string representation of the attribute value."""
        val = self.attrs[attr_name]
        return self.attr_def[attr_name].attr_str(val)


def sort_element(elements: list[CDGElement]) -> list[CDGElement]:
    """Sort CDG elements by their names."""
    return sorted(elements, key=lambda x: x.name)


class CDGNode(CDGElement):
    """Constrained Dynamic Graph (CDG) node class."""

    edges: set["CDGEdge"]
    reduction: Reduction
    _init_vals: list[float]
    _traces: list[np.ndarray]

    def __init__(self, cdg_type: "NodeType", name: str, **attrs) -> None:
        super().__init__(cdg_type, name, **attrs)
        self.edges = set()
        self._init_vals = [None for _ in range(cdg_type.order)]
        self._trace = [None for _ in range(cdg_type.order)]

    @property
    def init_vals(self) -> list[float]:
        """Access all intitial values of the node.

        Returns:
            list[float]: Initial values listed by order (0~cdg_type.order-1)
        """
        return self._init_vals

    @init_vals.setter
    def init_vals(self, vals: list[float]) -> None:
        node_order = self.cdg_type.order
        if len(vals) != node_order:
            raise RuntimeError(
                f"Invalid initial value length. Expect {node_order} values."
            )
        self._init_vals = vals

    def init_val(self, n: int) -> float:
        """Access the initial value of the n-th order deravative of the node.

        Args:
            n (int): the order of the initial value to access

        Returns:
            float: The n-th order intitial value
        """
        return self._init_vals[n]

    def set_init_val(self, val: float, n: int) -> None:
        """Set the initial value of the n-th order deravative of the node.

        Args:
            val (float): the initial value
            n (int): the order of the initial value to access
        """
        node_order = self.cdg_type.order
        if n >= node_order:
            raise RuntimeError(
                f"Invalid initial value order. Expect order < {node_order}."
            )
        self._init_vals[n] = val

    @property
    def degree(self) -> int:
        """Return the degree of the node."""
        return len(self.edges)

    def get_non_switchable(self) -> set["CDGEdge"]:
        """return all the non-switchable edges"""
        for edge in self.edges:
            if not edge.switchable:
                yield edge

    def get_switchable(self) -> set["CDGEdge"]:
        """return all the switchable edges"""
        for edge in self.edges:
            if edge.switchable:
                yield edge

    def add_edge(self, edge: "CDGEdge"):
        """Add the edge to the node."""
        self.edges.add(edge)

    def remove_edge(self, edge: "CDGEdge"):
        """Remove the edge from the node."""
        self.edges.remove(edge)

    def which_tgt(self, edge: "CDGEdge") -> Target:
        """Return whether this node is src/dst/self of the edge."""
        if edge.src == edge.dst:
            return SELF
        elif self.is_src(edge):
            return SRC
        elif self.is_dst(edge):
            return DST

    def is_src(self, edge: "CDGEdge") -> bool:
        """Return whether this node is the source of the edge."""
        return edge.src == self

    def is_dst(self, edge: "CDGEdge") -> bool:
        """Return whether this node is the destination of the edge."""
        return edge.dst == self

    def is_neighbor(self, node: "CDGNode") -> bool:
        """Return whether the node is a neighbor of this node."""
        for edge in self.edges:
            if self.get_neighbor(edge=edge) == node:
                return True

        return False

    def get_neighbor(self, edge: "CDGEdge"):
        """Return the neighbor of this node."""
        if self.is_src(edge):
            return edge.dst
        elif self.is_dst(edge):
            return edge.src
        else:
            assert False, f"{self} does not connect to {edge}"

    def print_local(self):
        """Print the local view of this node."""
        print(self.name)
        for edge in self.edges:
            if self.is_src(edge=edge):
                arrow = f"-{edge.name}>"
            else:
                arrow = f"<{edge.name}-"
            print("\t", arrow, self.get_neighbor(edge=edge).name)

    def get_trace(self, n: int) -> np.ndarray:
        """Access the trace of the n-th order state of the node from simulation.

        Args:
            n (int): the order of the state to access
        Returns:
            np.ndarray: the trace of the n-th order state
        """
        if n > self.cdg_type.order:
            raise RuntimeError("Invalid order")
        elif self._trace[n] is None:
            raise RuntimeError("Trace not available")
        return self._trace[n]

    def set_trace(self, n: int, trace: np.ndarray) -> None:
        """Set the trace of the node from simulation.

        Args:
            n (int): the order of the state to set
            trace (np.ndarray): the trace of the n-th order state
        """
        self._trace[n] = trace


class CDGEdge(CDGElement):
    """Constrained Dynamic Graph (CDG) edge class."""

    def __init__(self, cdg_type: "CDGType", name: str, **attrs) -> None:
        super().__init__(cdg_type, name, **attrs)
        self._src, self._dst = None, None

    def connect(self, src: CDGNode, dst: CDGNode) -> None:
        """Connect this edge to two nodes."""
        if self._src is not None or self._dst is not None:
            raise RuntimeError("Edge already connected")
        self._src, self._dst = src, dst

    @property
    def src(self) -> CDGNode:
        """Return the source node of this edge."""
        return self._src

    @property
    def dst(self) -> CDGNode:
        """Return the destination node of this edge."""
        return self._dst

    @property
    def switchable(self) -> bool:
        """Return whether this edge is also a switch."""
        return self._switchable

    @switchable.setter
    def switchable(self, switchable: bool) -> None:
        self._switchable = switchable


class CDG:
    """
    Constrained Dynamic Graph (CDG) class.
    """

    _order_to_nodes: list[dict]

    def __init__(self) -> None:
        self._order_to_nodes = [set()]
        self._edges = set()
        self._switches = set()

    def connect(self, edge: CDGEdge, src: CDGNode, dst: CDGNode):
        """Add an edge to the graph."""
        edge.connect(src=src, dst=dst)
        src.add_edge(edge)
        dst.add_edge(edge)
        self._add_node(src)
        self._add_node(dst)
        self._edges.add(edge)
        if edge.switchable:
            self._switches.add(edge)

    def delete_node(self, node: CDGNode) -> None:
        """Delete a node from the graph.
        Disconnect edges connected to the node accordingly.
        """
        order = node.order
        self._order_to_nodes[order].remove(node)
        raise NotImplementedError

    def check_exist(self, element: CDGElement) -> bool:
        """Check if an element exists in the graph."""
        if isinstance(element, CDGNode):
            order = element.order
            return order < self.ds_order and element in self._order_to_nodes[order]
        elif isinstance(element, CDGEdge):
            return element in self._edges

    def check_connectivity(self) -> bool:
        """Check if the graph is well-connected."""

        raise NotImplementedError

    def _add_node(self, node: CDGNode) -> None:
        """Add a node to the graph."""
        order = node.order
        max_order = len(self._order_to_nodes) - 1
        if order > max_order:
            self._order_to_nodes += [set() for _ in range(order - max_order)]
        self._order_to_nodes[order].add(node)

    def nodes_in_order(self, order: int) -> list[CDGNode]:
        """Return nodes in the graph based on order sorted by node.name.

        If order is -1, return all nodes.
        """
        if order == -1:
            return sort_element(list(set.union(*self._order_to_nodes)))
        else:
            return sort_element(list(self._order_to_nodes[order]))

    def stateful_nodes(self) -> list[CDGNode]:
        """Access all stateful nodes, i.e., nodes with order > 0.

        Returns:
            list[CDGNode]: list of stateful nodes sorted by name
        """

        nodes = []
        for order in range(1, self.ds_order + 1):
            nodes += self.nodes_in_order(order)
        return nodes

    def total_1st_order_states(self) -> int:
        """The total number of state variables in the system of
        1st order differential equations.

        Returns:
            int: the number of state variables
        """
        return sum(
            [
                len(self.nodes_in_order(order)) * order
                for order in range(1, self.ds_order + 1)
            ]
        )

    def initialize_all_states(self, val: float = None, rand: bool = False) -> None:
        """Initialize all state variables in the system of
        1st order differential equations.

        The function will set all the initial values of the stateful nodes with val or
        random values if rand is True.

        Args:
            val (float): the initial value
            rand (bool): whether to initialize with random values
        """
        if val and rand:
            raise RuntimeError("Cannot specify both val and rand.")
        for node in self.stateful_nodes():
            node.init_vals = [
                np.random.rand() if rand else val for _ in range(node.order)
            ]

    @property
    def nodes(self) -> list[CDGNode]:
        """Return all nodes in the graph sorted by node.name."""
        return sort_element(list(set.union(*self._order_to_nodes)))

    @property
    def edges(self) -> list[CDGEdge]:
        """Return all edges in the graph sorted by node.name."""
        return sort_element(list(self._edges))

    @property
    def switches(self) -> list[CDGElement]:
        """Return all switches in the graph sorted by node.name."""
        return sort_element(list(self._switches))

    @property
    def ds_order(self) -> int:
        """Order of the system of differential equations."""
        return len(self._order_to_nodes) - 1
