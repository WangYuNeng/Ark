from ark.globals import Direction
from ark.specification.types import CDGType, StatefulNodeType, NodeType, EdgeType

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
    def src(self) -> list:
        return self._src

    @property
    def dst(self) -> list:
        return self._dst

class CDG:

    def __init__(self) -> None:
        self._stateful_nodes = []
        self._stateless_nodes = []
        self._edges = []
        self._switches = []
        self._elements = []

    def add_node(self, name: str, cdg_type: CDGType, attrs: dict) -> CDGNode:
        id = len(self._elements)
        node = CDGNode(id=id, name=name, cdg_type=cdg_type, attrs=attrs)
        if isinstance(cdg_type, StatefulNodeType):
            self._stateful_nodes.append(node)
        elif isinstance(cdg_type, NodeType):
            self._stateless_nodes.append(node)
        self._elements.append(node)
        return node

    def add_edge(self, name: str, cdg_type: CDGType, attrs: dict, src: CDGNode, dst: CDGNode) -> CDGNode:
        id = len(self._elements)
        edge = CDGEdge(id=id, name=name, cdg_type=cdg_type, attrs=attrs, src=src, dst=dst)
        src.add_edge(edge)
        dst.add_edge(edge)
        self._edges.append(edge)
        self._elements.append(edge)
        return edge

    @property
    def nodes(self) -> list:
        return self._stateful_nodes + self.stateless_nodes
    @property
    def stateful_nodes(self) -> list:
        return self._stateful_nodes

    @property
    def stateless_nodes(self) -> list:
        return self._stateless_nodes
    
    @property
    def edges(self) -> list:
        return self._edges

    @property
    def switches(self) -> list:
        return self._switches