from abc import ABC, abstractmethod

class Edge(ABC):

    def __init__(self, name) -> None:
        self._name = name
        self._src, self._dst = None, None

    def connect(self, src, dst):
        self._src, self._dst = src, dst

    @property
    def name(self) -> str:
        return self._name
    
    def linked_name(self, n) -> str:
        if n == self.src:
            return self.dst.name
        elif n == self.dst:
            return self.src.name
        else:
            assert False, 'edge {} does not connect to node {}'.format(self.name, n.name)

    def linked_spice_name(self, n) -> str:
        if n == self.src:
            return self.dst.name
        elif n == self.dst:
            return self.src.name
        else:
            assert False, 'edge {} does not connect to node {}'.format(self.name, n.name)


    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @abstractmethod
    def validation(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dynamical_system(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_spice(self) -> str:
        raise NotImplementedError