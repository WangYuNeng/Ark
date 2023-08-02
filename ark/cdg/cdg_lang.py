from ark.specification.cdg_types import NodeType, EdgeType

class CDGLang:

    def __init__(self,name, inherits=None):
        self.name = name   
        self.inherits = inherits

        self._node_types = {}
        self._node_order = []

        self._edge_types = {}
        self._edge_order = []

        self._production_rules= {}
        self._validation_rules = {}

    def edge_types(self):
        if not self.inherits is None:
            for n in self.inherits.edge_types():
                yield n


        for n in self._edge_order:
            yield self._edge_types[n]

    def validation_rules(self):
        for valid in self._validation_rules.values():
            for rule in valid:
                yield rule


    def production_rules(self):
        for prod in self._production_rules.values():
            yield prod 

    def is_inherited(self,val):
        if val in self.inherits.node_types():
            return True
        
        if val in self.inherits.edge_types():
            return True

        return False

    def node_types(self):
        if not self.inherits is None:
            for n in self.inherits.node_types():
                yield n

        for n in self._node_order:
            yield self._node_types[n]

    def add_types(self,*args):
        for arg in args:
            if isinstance(arg, EdgeType):
                assert(not arg.name in self._edge_types)
                self._edge_types[arg] = arg
                self._edge_order.append(arg)

            elif isinstance(arg, NodeType):
                assert(not arg.name in self._node_types)
                self._node_types[arg] = arg
                self._node_order.append(arg)

            else:
                raise Exception("unknown type: %s" % arg)

    def add_validation_rules(self,*args):
        for arg in args:
            if not arg.tgt_node_type  in self._validation_rules:
                self._validation_rules[arg.tgt_node_type] = []

            self._validation_rules[arg.tgt_node_type].append(arg)


    def add_production_rules(self,*args):
        for arg in args:
            self._production_rules[arg.identifier] = arg

    def filename(self,suffix,extension):
        def format_filename(text):
            text = text.lower()
            return "-".join(text.split(" "))

        prefix = format_filename(self.name)
        suffix = format_filename(suffix)
        filename = "lang_%s_%s.%s" % (prefix, suffix, extension)
        return filename

