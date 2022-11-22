class CDGNode:

    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name
        self._edges = []

    @property
    def id(self) -> int:
        return self._id
        
    @property
    def name(self)-> str:
        return self._name
    
    @property
    def edges(self) -> list:
        return self._edges

class CDGEdge:

    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name
        self._src = None
        self._dst = None

    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self)-> str:
        return self._name
        
    @property
    def src(self) -> list:
        return self._src

    @property
    def dst(self) -> list:
        return self._dst

    def is_src(self, node: CDGNode) -> bool:
        return node == self.src

    def is_dst(self, node: CDGNode) -> bool:
        return node == self.dst

class CDG:

    def __init__(self) -> None:
        self._stateful_nodes = []
        self._stateless_nodes = []
        self._edges = []
        self._switches = []

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