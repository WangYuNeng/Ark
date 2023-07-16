from typing import Mapping
from ark.globals import Direction
from ark.specification.generation_rule import SRC, DST, SELF, GenRuleKeyword
from ark.specification.attribute_def import AttrDef, AttrImpl
from ark.reduction import Reduction
# from ark.specification.types import CDGType, NodeType, EdgeType

class CDGElement:
    """Base class for CDG nodes and edges."""
    attr_def: Mapping[str, AttrDef]

    def __init__(self, cdg_type: "CDGType", name: str, **attrs: Mapping[str, AttrImpl]) -> None:
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

    reduction: Reduction

    def __init__(self, cdg_type: "NodeType", name: str, **attrs) -> None:
        super().__init__(cdg_type, name, **attrs)
        self.edges = set()

    @property
    def degree(self) -> int:
        return len(self.edges)

    def add_edge(self, e):
        self.edges.add(e)
    
    def remove_edge(self, e):
        self.edges.remove(e)

    def gen_tgt_type(self, edge: "CDGEdge") -> GenRuleKeyword:
        """Return whether this node is src/dst/self of the edge."""
        if edge.src == edge.dst:
            return SELF
        elif edge.src == self:
            return SRC
        elif edge.dst == self:
            return DST

    def is_src(self, edge: "CDGEdge") -> bool:
        return edge.src == self

    def is_dst(self, edge: "CDGEdge") -> bool:
        return edge.dst == self

    def is_neighbor(self, node: "CDGNode") -> bool:
        for edge in self.edges:
            if self.get_neighbor(edge=edge) == node:
                return True

        return False

    def get_direction(self, edge: "CDGEdge") -> int:
        if self.is_src(edge) and self.is_dst(edge):
            return Direction.SELF
        elif self.is_src(edge):
            return Direction.OUT
        elif self.is_dst(edge):
            return Direction.IN
        else:
            assert False, f'{self} does not connect to {edge}'

    def get_neighbor(self, edge: 'CDGEdge'):
        if self.is_src(edge):
            return edge.dst
        elif self.is_dst(edge):
            return edge.src
        else:
            assert False, f'{self} does not connect to {edge}'

    def print_local(self):
        print(self.name)
        for edge in self.edges:
            if self.is_src(edge=edge):
                arrow = f'-{edge.name}>'
            else:
                arrow = f'<{edge.name}-'
            print('\t', arrow, self.get_neighbor(edge=edge).name)

class CDGEdge(CDGElement):
    """Constrained Dynamic Graph (CDG) edge class."""

    def __init__(self, cdg_type: "CDGType", name: str, **attrs) -> None:
        super().__init__(cdg_type, name, **attrs)
        self._src, self._dst = None, None

    def connect(self, src: CDGNode, dst: CDGNode) -> None:
        """Connect this edge to two nodes."""
        if self._src is not None or self._dst is not None:
            raise RuntimeError('Edge already connected')
        self._src, self._dst = src, dst

    @property
    def src(self) -> CDGNode:
        """Return the source node of this edge."""
        return self._src

    @property
    def dst(self) -> CDGNode:
        """Return the destination node of this edge."""
        return self._dst

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
        edge.connect(src, dst)
        src.add_edge(edge)
        dst.add_edge(edge)
        self._add_node(src)
        self._add_node(dst)
        self._edges.add(edge)

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

    @property
    def nodes(self) -> list[CDGNode]:
        """Return all nodes in the graph sorted by node.name."""
        return sort_element(list(set.union(*self._order_to_nodes)))

    @property
    def edges(self) -> list[CDGEdge]:
        """Return all edges in the graph."""
        return list(self._edges)

    @property
    def switches(self) -> list[CDGElement]:
        raise NotImplementedError

    @property
    def ds_order(self) -> int:
        """Order of the system of differential equations."""
        return len(self._order_to_nodes) - 1
