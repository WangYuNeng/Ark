from ark.test.ladder.spicemodel import *
from ark.test.ladder.simulator import Simulator
from ark.cdg.cdg import CDG

class Generator:

    def __init__(self) -> None:
        self.spice_model = LadderGraph()
        pass

    def produce_vn(self, node):
        pass

    def produce_iv(self):
        pass