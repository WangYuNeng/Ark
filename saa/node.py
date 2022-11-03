from abc import ABC, abstractmethod

class Node(ABC):

    def __init__(self, name, has_state) -> None:
        self._name = name
        self._has_state = has_state
        self._conn = []

    def add_conn(self, edge):
        self._conn.append(edge)

    @property
    def has_state(self) -> bool:
        return self._has_state

    @property
    def conn(self) -> list:
        return self._conn

    @property
    def name(self) -> str:
        return self._name

    @property
    def ddt_name(self) -> str:
        return f'ddt_{self.name}'
    
    @property
    def spice_name(self) -> str:
        return self._name

    def __eq__(self, n2) -> bool:
        return n2.name == self.name

    @abstractmethod
    def validation(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dynamical_system(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_spice(self) -> str:
        raise NotImplementedError

class StatefulNode(Node):

    def __init__(self, name) -> None:
        super().__init__(name, has_state=True)

class StatelessNode(Node):

    def __init__(self, name) -> None:
        super().__init__(name, has_state=False)