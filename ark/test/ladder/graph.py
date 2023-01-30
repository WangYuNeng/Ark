from abc import ABC, abstractmethod
from ark.test.ladder.node import Node
from ark.test.ladder.edge import Edge

class Graph(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_node(self, node_type, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_edge(self, edge_type, **kwargs):
        raise NotImplementedError

    def connect(self, edge: Edge, src: Node, dst: Node):
        edge.connect(src=src, dst=dst)
        src.add_conn(edge=edge)
        dst.add_conn(edge=edge)

    @abstractmethod
    def to_dynamical_system(self):
        raise NotImplementedError

    @abstractmethod
    def to_spice(self):
        raise NotImplementedError