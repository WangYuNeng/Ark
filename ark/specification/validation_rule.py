import numpy as np
from ark.cdg.cdg import CDGNode, CDGEdge
from ark.specification.types import EdgeType, NodeType, StatefulNodeType

class DegreeConstraint:

    def __init__(self, expr: str) -> None:
        self._expr = expr

    @property
    def expr(self):
        return self._expr

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.expr)

class Connection:

    def __init__(self, edge_type: EdgeType, direction: int, degree: DegreeConstraint, node_types: list) -> None:
        self._edge_type = edge_type
        self._direction = direction
        self._degree = degree
        self._node_types = node_types

    @property
    def edge_type(self):
        return self._edge_type
    
    @property
    def direction(self):
        return self._direction

    @property
    def degree(self):
        return self._degree
    
    @property
    def node_types(self):
        return self._node_types

    @property
    def identifiers(self) -> set:
        nt: NodeType

        ids = set()
        for nt in self.node_types:
            ids.add(self.get_identifier(edge_type=self.edge_type, direction=self.direction, node_type=nt))
        return ids

    @staticmethod
    def get_identifier(edge_type: EdgeType, direction: int, node_type: NodeType):
        return repr([edge_type.type_name, direction, node_type.type_name])

    def __repr__(self) -> str:
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.edge_type, self.direction, self.degree, self.node_types)

class ValRule:

    def __init__(self, tgt_node_type: NodeType, connections: list) -> None:
        self._tgt_node_type = tgt_node_type
        self._connections = connections

    @property
    def tgt_node_type(self):
        return self._tgt_node_type

    @property
    def connections(self):
        return self._connections

    def get_validation_matrix(self, node: CDGNode):
        edge: CDGEdge
        conn: Connection

        matrix = np.zeros(shape=(node.degree, len(self._connections)))
        for i, edge in enumerate(node.edges):
            edge_id = Connection.get_identifier(edge_type=edge.cdg_type, direction=node.get_direction(edge), node_type=node.get_neighbor(edge).cdg_type)
            for j, conn in enumerate(self._connections):
                conn_ids = conn.identifiers
                if edge_id in conn_ids:
                    matrix[i, j] = 1
        
        print(matrix)
        print([conn.degree for conn in self._connections])


    def __repr__(self) -> str:
        return '{}({} {})'.format(self.__class__.__name__, self.tgt_node_type, self.connections)