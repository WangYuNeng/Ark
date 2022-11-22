class CDGType:

    def __init__(self, name: str, type_name: str, attr_name: list, attr_range: list) -> None:
        self._name = name
        self._type_name = type_name
        self._attr_name = attr_name
        self._attr_range = attr_range
    
    @property
    def name(self):
        return self._name
    
    @property
    def type_name(self):
        return self._type_name
    
    @property
    def attr_name(self):
        return self._attr_name
    
    @property
    def attr_range(self):
        return self._attr_range
    
    def __repr__(self) -> str:
        return '{}({} {} {} {})'.format(self.name, self.type_name, self.attr_name, self.attr_range)

class NodeType(CDGType):

    def __init__(self, type_name: str, attr_name: list, attr_range: list) -> None:
        super().__init__('NodeType', type_name, attr_name, attr_range)

class EdgeType(CDGType):

    def __init__(self, type_name: str, attr_name: list, attr_range: list) -> None:
        super().__init__('EdgeType', type_name, attr_name, attr_range)
    
class StatefulNodeType(NodeType):

    def __init__(self, type_name: str, attr_name: list, attr_range: list) -> None:
        CDGType.__init__(self, 'StatefulNodeType', type_name, attr_name, attr_range)
