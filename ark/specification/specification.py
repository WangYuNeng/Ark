import ast
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.generation_rule import GenRule
from ark.specification.validation_rule import ValRule
from ark.cdg.cdg import CDG, CDGEdge, CDGNode

class CDGSpec:

    def __init__(self, cdg_types: list, generation_rules: list, validation_rules: list):
        self._cdg_types = cdg_types
        self._generation_rules = generation_rules
        self._gen_rule_dict = self._collect_gen_identifier()
        self._gen_rule_class = self._check_class(generation_rules)
        self._validation_rules = validation_rules
        self._val_rule_dict = self._collect_val_identifier()
        self._val_rule_class = self._check_class(validation_rules)

    @property
    def cdg_types(self):
        return self._cdg_types

    @property
    def generation_rules(self):
        return self._generation_rules

    @property
    def gen_rule_dict(self):
        return self._gen_rule_dict

    @property
    def gen_rule_class(self):
        return self._gen_rule_class

    @property
    def validation_rules(self):
        return self._validation_rules

    @property
    def val_rule_dict(self):
        return self._val_rule_dict

    @property
    def val_rule_class(self):
        return self._val_rule_class    

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
        return val_type_dict

    def _collect_gen_identifier(self):
        rule: GenRule

        rules = self.generation_rules
        genid_dict = {
            rule.identifier: rule for rule in rules
        }
        return genid_dict

    def _check_class(self, instances: list):

        class_obj = instances[0].__class__
        for instance in instances[1:]:
            if instance.__class__ != class_obj:
                assert False, 'Rules should all belong to the same class!'
        return class_obj
