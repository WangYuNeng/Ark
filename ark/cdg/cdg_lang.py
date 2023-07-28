from ark.specification.cdg_types import NodeType, EdgeType

class CDGLang:

    def __init__(self,name, inherits=None):
        self.name = name   
        self.inherits = inherits

        self._node_types = {}
        self._node_order = []

        self._edge_types = {}
        self._edge_order = []

        self._relations = {}
        self._validation_rules = {}

    def edge_types(self):
        for n in self._edge_order:
            yield self._edge_types[n]


    def node_types(self):
        for n in self._node_order:
            yield self._node_types[n]

    def add_types(self,*args):
        for arg in args:
            if isinstance(arg, EdgeType):
                assert(not arg.name in self._edge_types)
                self._edge_types[arg.name] = arg
                self._edge_order.append(arg.name)

            elif isinstance(arg, NodeType):
                assert(not arg.name in self._node_types)
                self._node_types[arg.name] = arg
                self._node_order.append(arg.name)

            else:
                raise Exception("unknown type: %s" % arg)


