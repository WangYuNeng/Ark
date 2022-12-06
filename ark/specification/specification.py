import ast
from ark.specification.types import EdgeType, NodeType, StatefulNodeType
from ark.specification.generation_rule import GenRule
from ark.specification.validation_rule import ValRule
from ark.cdg.cdg import CDG, CDGEdge, CDGNode

class CDGSpec:

    def __init__(self, cdg_types: list, generation_rules: list, validation_rules: list):
        self.cdg_types = cdg_types
        self.generation_rules = generation_rules
        self.validation_rules = validation_rules

        self._collect_val_identifier()
        self._collect_gen_identifier()

    def get_candidate_val_rules(self, node: CDGNode):
        return self._val_type_dict[node.cdg_type.type_name]

    def apply_gen_rule(self, edge: CDGEdge, src: CDGNode, dst: CDGNode, tgt: bool):
        '''
        Call the corresponding generation rule to walk the ast and generate expr
        '''
        id = GenRule.get_identifier(tgt_et=edge.cdg_type, src_nt=src.cdg_type, dst_nt=dst.cdg_type, gen_tgt=tgt)
        rule = self._genid_dict[id]
        return rule.apply(edge=edge, src_node=src, dst_node=dst)

    def _collect_val_identifier(self):
        rule: ValRule
        val_type_dict = {}
        for rule in self.validation_rules:
            type_name = rule.tgt_node_type.type_name
            if type_name in val_type_dict:
                val_type_dict[type_name].append(rule)
            else:
                val_type_dict[type_name] = [rule]
        self._val_type_dict = val_type_dict

    def _collect_gen_identifier(self):
        rule: GenRule

        rules = self.generation_rules
        self._genid_dict = {
            rule.identifier: rule for rule in rules
        }
