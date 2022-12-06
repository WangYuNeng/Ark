class CDGTypeDefs:
    NOTETYPE = 'NodeType'
    EDGETYPE = 'EdgeType'
    STATEFULNODETYPE = 'StatefulNodeType'

class CDGType:

    def __init__(self, name: str, type_name: str, attrs: dict) -> None:
        self._name = name
        self._type_name = type_name
        self._attrs = attrs
    
    @property
    def name(self):
        return self._name
    
    @property
    def type_name(self):
        return self._type_name
    
    @property
    def attrs(self):
        return self._attrs

    @property
    def attr_name(self):
        return self._attrs.keys()
    
    def __repr__(self) -> str:
        return '{}({}, {})'.format(self.name, self.type_name, self.attrs)

class NodeType(CDGType):

    def __init__(self, type_name: str, attrs: dict) -> None:
        super().__init__(CDGTypeDefs.NOTETYPE, type_name, attrs)

class EdgeType(CDGType):

    def __init__(self, type_name: str, attrs: dict) -> None:
        super().__init__(CDGTypeDefs.EDGETYPE, type_name, attrs)
    
class StatefulNodeType(NodeType):

    def __init__(self, type_name: str,  attrs: dict) -> None:
        CDGType.__init__(self, CDGTypeDefs.STATEFULNODETYPE, type_name, attrs)