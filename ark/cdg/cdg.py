from ark.globals import Direction
from ark.specification.types import CDGType, StatefulNodeType, NodeType, EdgeType
from collections import OrderedDict

class CDGElement:

    def __init__(self, id: int, name: str, cdg_type: CDGType, attrs: dict) -> None:
        self._id = id
        self._name = name
        self._cdg_type = cdg_type
        self._attrs = attrs

    @property
    def id(self) -> int:
        return self._id
        
    @property
    def name(self)-> str:
        return self._name

    @property
    def cdg_type(self) -> CDGType:
        return self._cdg_type

    @property
    def attrs(self) -> dict:
        return self._attrs

    def __repr__(self) -> str:
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self._id, self.name, self.cdg_type, self.attrs)

    def __eq__(self, __o: object) -> bool:
        return self._id == __o.id
    
    def __hash__(self) -> int:
        return self.id.__hash__()


class CDGNode(CDGElement):

    def __init__(self, id: int, name: str, cdg_type: CDGType, attrs: dict) -> None:
        super().__init__(id, name, cdg_type, attrs)
        self._edges = set()

    @property
    def edges(self) -> list:
        return self._edges

    @property
    def degree(self) -> int:
        return len(self._edges)

    def add_edge(self, e):
        self._edges.add(e)

    def is_src(self, edge: 'CDGEdge') -> bool:
        return edge.src == self

    def is_dst(self, edge: 'CDGEdge') -> bool:
        return edge.dst == self

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


class CDGEdge(CDGElement):

    def __init__(self, id: int, name: str, cdg_type: CDGType, attrs: dict, src: CDGNode, dst: CDGNode) -> None:
        super().__init__(id, name, cdg_type, attrs)
        self._src = src
        self._dst = dst
        
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
    
        # TODO: SHOULD WE THROW ERROR IF NAME IS ALREADY IN DICTIONARY?
    def add_node(self, name: str, cdg_type: CDGType, attrs: dict) -> CDGNode:
        id = self._get_id()
        node = CDGNode(id=id, name=name, cdg_type=cdg_type, attrs=attrs)
        if isinstance(cdg_type, StatefulNodeType):
            self._stateful_nodes[name] = node
        elif isinstance(cdg_type, NodeType):
            self._stateless_nodes[name] = node
        self._elements[name] = node
        return node

    def add_edge(self, name: str, cdg_type: CDGType, attrs: dict, src: CDGNode, dst: CDGNode) -> CDGEdge:
        id = self._get_id()
        edge = CDGEdge(id=id, name=name, cdg_type=cdg_type, attrs=attrs, src=src, dst=dst)
        src.add_edge(edge)
        dst.add_edge(edge)
        self._edges[name] = edge
        self._elements[name] = edge
        return edge

    def delete_node(self, node_name: str) -> None: 
        node = self._elements[node_name]
        # delete edges
        for edge in node.edges():
            self._edges.pop(edge.name())
        # verify there are no edges connected 
        if self._verify_no_edges(node_name):
            if isinstance(node.cdg_type(), StatefulNodeType):
                self._stateful_nodes.pop(node.name())
            elif isinstance(node.cdg_type(), NodeType):
                self._stateless_nodes.pop(node.name())
            self._elements.pop(node.name())
        else:
            print("edges still connected to node")
        
    def _verify_no_edges(self, node_name: str) -> bool:
        raise NotImplementedError

    @property
    def nodes(self) -> list:
        return list(self._stateful_nodes.values()) + list(self.stateless_nodes.values())
    @property
    def stateful_nodes(self) -> list:
        return list(self._stateful_nodes.values())

    @property
    def stateless_nodes(self) -> list:
        return list(self.stateless_nodes.values())
    
    @property
    def edges(self) -> list:
        return list(self._edges.values())

    @property
    def switches(self) -> list:
        return self._switches