import ast
from ark.cdg.cdg import CDG, CDGEdge, CDGNode

class CDGSpec:

    def __init__(self):
        self.cdg_types = None
        self.generation_rules = None
        self.validation_rules = None

    def apply_gen_rule(self, edge: CDGEdge, src: CDGNode, dst: CDGNode, tgt: bool):
        '''
        Walk the ast and generate expr
        '''
        pass