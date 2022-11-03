from saa.node import Node
from saa.edge import Edge


def connect(edge: Edge, src: Node, dst: Node):
    edge.connect(src=src, dst=dst)
    src.add_conn(edge=edge)
    dst.add_conn(edge=edge)
