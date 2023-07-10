from collections import OrderedDict
import warnings
from ark.globals import Direction
# from ark.specification.types import CDGType, NodeType, EdgeType

class CDGElement:

    def __init__(self, cdg_type: 'CDGType', **attrs) -> None:
        self.cdg_type = cdg_type
        self.attrs = attrs


class CDGNode(CDGElement):

    def __init__(self, cdg_type: 'NodeType', **attrs) -> None:
        super().__init__(cdg_type, **attrs)
        self._edges = set()

    @property
    def edges(self) -> list:
        return list(self._edges)

    @property
    def degree(self) -> int:
        return len(self._edges)

    def add_edge(self, e):
        self._edges.add(e)
    
    def remove_edge(self, e):
        self._edges.remove(e)

    def is_src(self, edge: 'CDGEdge') -> bool:
        return edge.src == self

    def is_dst(self, edge: 'CDGEdge') -> bool:
        return edge.dst == self

    def is_neighbor(self, node: 'CDGNode') -> bool:
        for edge in self.edges:
            if self.get_neighbor(edge=edge) == node:
                return True

        return False

    def get_direction(self, edge: 'CDGEdge') -> int:
        if self.is_src(edge) and self.is_dst(edge):
            return Direction.SELF
        elif self.is_src(edge):
            return Direction.OUT
        elif self.is_dst(edge):
            return Direction.IN
        else:
            assert False, '{} does not connect to {}'.format(self, edge)

    def get_neighbor(self, edge: 'CDGEdge'):
        if self.is_src(edge):
            return edge.dst
        elif self.is_dst(edge):
            return edge.src
        else:
            assert False, '{} does not connect to {}'.format(self, edge)

    def print_local(self):
        print(self.name)
        for edge in self.edges:
            if self.is_src(edge=edge):
                arrow = '-{}>'.format(edge.name)
            else:
                arrow = '<{}-'.format(edge.name)
            print('\t', arrow, self.get_neighbor(edge=edge).name)
            


class CDGEdge(CDGElement):

    def __init__(self, id: int, name: str, cdg_type: 'CDGType', attrs: dict, src: CDGNode, dst: CDGNode) -> None:
        super().__init__(id, name, cdg_type, attrs)
        self._src = src
        self._dst = dst
        
    def set_src(self, node: CDGNode) -> None:
        self._src = node
        
    def set_dst(self, node: CDGNode) -> None:
        self._dst = node
        
    @property
    def src(self) -> CDGNode:
        return self._src

    @property
    def dst(self) -> CDGNode:
        return self._dst

class CDG:

    def __init__(self) -> None:
        self._stateful_nodes = OrderedDict()
        self._stateless_nodes = OrderedDict()
        self._edges = OrderedDict()
        self._switches = []
        self._elements = OrderedDict()
        self._next_id = 0
        
    def _get_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1
    
    def add_node(self, name: str, cdg_type: 'CDGType', attrs: dict) -> CDGNode:
        id = self._get_id()
        node = CDGNode(id=id, name=name, cdg_type=cdg_type, attrs=attrs)
        if isinstance(cdg_type, StatefulNodeType):
            self._stateful_nodes[id] = node
        elif isinstance(cdg_type, NodeType):
            self._stateless_nodes[id] = node
        self._elements[id] = node
        return node

    def add_edge(self, name: str, cdg_type: 'CDGType', attrs: dict, src: CDGNode, dst: CDGNode) -> CDGEdge:
        id = self._get_id()
        edge = CDGEdge(id=id, name=name, cdg_type=cdg_type, attrs=attrs, src=src, dst=dst)
        src.add_edge(edge)
        dst.add_edge(edge)
        self._edges[id] = edge
        self._elements[id] = edge
        return edge
    
    def update_edge(self, edge: CDGEdge):
        id = edge.id
        self._edges[id] = edge
        self._elements[id] = edge
        
    def delete_node(self, node_id: int) -> None: 
        node = self._elements[node_id]
        if isinstance(node.cdg_type, StatefulNodeType):
            self._stateful_nodes.pop(node.id)
        elif isinstance(node.cdg_type, NodeType):
            self._stateless_nodes.pop(node.id)
        self._elements.pop(node.id)
        
    def check_connectivity(self) -> bool:
        for e in self.edges:
            if e.src.id not in self._elements:
                warnings.warn("Source Node of Edge not in graph")
            elif e.dst.id not in self._elements:
                warnings.warn("Source Node of Edge not in graph")
        return True

    @property
    def nodes(self) -> list:
        return list(self._stateful_nodes.values()) + list(self._stateless_nodes.values())
    
    @property
    def stateful_nodes(self) -> list:
        return list(self._stateful_nodes.values())

    @property
    def stateless_nodes(self) -> list:
        return list(self._stateless_nodes.values())
    
    @property
    def edges(self) -> list:
        return list(self._edges.values())

    @property
    def switches(self) -> list:
        return self._switches