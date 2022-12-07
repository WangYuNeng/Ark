import ast
import copy

from ark.rewrite import RewriteGen
from ark.cdg.cdg import CDG, CDGNode, CDGEdge
from ark.specification.specification import CDGSpec
from ark.specification.generation_rule import GenRule

class ArkCompiler():

    def __init__(self, rewrite: RewriteGen) -> None:
        self._rewrite = rewrite
        pass

    def compile(self, cdg: CDG, cdg_spec: CDGSpec):

        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: GenRule

        stmts = []
        rule_dict = cdg_spec.gen_rule_dict
        rule_class = cdg_spec.gen_rule_class
        for node in cdg.stateful_nodes:
            print(node)
            for edge in node.edges:
                src, dst = edge.src, edge.dst
                id = rule_class.get_identifier(tgt_et=edge.cdg_type, src_nt=src.cdg_type, dst_nt=dst.cdg_type, gen_tgt=node.is_dst(edge))
                gen_rule = rule_dict[id]
                self._rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                rhs = self._apply_rule(edge=edge, rule=rule_dict[id], transformer=self._rewrite)
                print('\t', edge)
                print('\t', ast.unparse(rhs))
    

    def _apply_rule(self, edge: CDGEdge, rule: GenRule, transformer: RewriteGen):
        gen_ast = copy.deepcopy(rule.fn_ast)
        transformer.visit(gen_ast)
        return gen_ast