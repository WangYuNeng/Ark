import ast

from ark.cdg.cdg import CDG
from ark.specification.specification import CDGSpec

class ArkCompiler():

    def __init__(self):
        pass

    def compile(self, cdg: CDG, cdg_spec: CDGSpec):
        if self.verify(cdg, cdg_spec):
            return self.generate(cdg, cdg_spec)
        return False

    def verify(self, cdg: CDG, cdg_spec: CDGSpec):
        pass

    def generate(self, cdg: CDG, cdg_spec: CDGSpec):
        stmts = []
        for node in cdg.stateful_nodes:
            for edge in node.edges:
                rhs = cdg_spec.apply_gen_rule(edge=edge, src=edge.src, dst=edge.dst, tgt=edge.is_dst(node))
