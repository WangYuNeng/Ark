from itertools import product
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.generation_rule import GenRule, GenRuleKeyword, GenRuleId
from ark.specification.validation_rule import ValRule
from ark.cdg.cdg import CDGEdge, CDGNode

class CDGSpec:

    def __init__(self, cdg_types: list[NodeType], generation_rules: list[GenRule], validation_rules: list[ValRule]):
        self._cdg_types = cdg_types
        self._generation_rules = generation_rules
        self._gen_rule_dict = self._collect_gen_identifier()
        self._gen_rule_class = self._check_class(generation_rules)
        if validation_rules is not None:
            self._validation_rules = validation_rules
            self._val_rule_dict = self._collect_val_identifier()
            self._val_rule_class = self._check_class(validation_rules)

    @property
    def cdg_types(self) -> list[NodeType]:
        """Access CDG types in the spec."""
        return self._cdg_types

    @property
    def generation_rules(self) -> list[GenRule]:
        """Access generation rules in the spec."""
        return self._generation_rules

    @property
    def gen_rule_dict(self) -> dict[GenRuleId, GenRule]:
        """Access mapping from generation rule identifier to generation rule."""
        return self._gen_rule_dict

    @property
    def gen_rule_class(self) -> GenRule:
        """Access the class of generation rules.
        
        TODO: Do we need this?"""
        return self._gen_rule_class

    @property
    def validation_rules(self) -> list[ValRule]:
        """Access validation rules in the spec."""
        return self._validation_rules

    @property
    def val_rule_dict(self) -> dict[str, ValRule]:
        """Access mapping from validation rule identifier to validation rule."""
        return self._val_rule_dict

    @property
    def val_rule_class(self) -> ValRule:
        """Access the class of validation rules.

        TODO: Do we need this?"""
        return self._val_rule_class 

    def match_gen_rule(self, edge: CDGEdge, src: CDGNode, dst: CDGNode, tgt: GenRuleKeyword) -> GenRule:
        '''
        Find the generation rule that matches the given edge, source node, destination node, and rule target.

        TODO: Handle multiple matches.
        '''
        def check_match(et: EdgeType, src_nt: NodeType, dst_nt: NodeType) -> GenRule | None:
            rule_id = GenRuleId(et, src_nt, dst_nt, tgt)
            if rule_id in self._gen_rule_dict:
                return self._gen_rule_dict[rule_id]
            return None

        et: EdgeType = edge.cdg_type
        src_nt: NodeType = src.cdg_type
        dst_nt: NodeType = dst.cdg_type

        match = check_match(et, src_nt, dst_nt)
        if match is not None:
            return match

        et_base = et.base_cdg_types()
        src_nt_base = src_nt.base_cdg_types()
        dst_nt_base = dst_nt.base_cdg_types()

        for et, src_nt, dst_nt in product(et_base, src_nt_base, dst_nt_base):
            match = check_match(et, src_nt, dst_nt)
            if match is not None:
                if et == et_base[-1] and src_nt == src_nt_base[-1] and dst_nt == dst_nt_base[-1]:
                    return match
                raise NotImplementedError('Have not implemented match in the heirarchy.')

        raise KeyError(f'No generation rule found for edge {edge}, \
                        source node {src}, destination node {dst}, and target {tgt}.')

    def _collect_val_identifier(self) -> dict[str, ValRule]:
        rule: ValRule
        val_type_dict = {}
        for rule in self.validation_rules:
            type_name = rule.tgt_node_type.type_name
            if type_name in val_type_dict:
                val_type_dict[type_name].append(rule)
            else:
                val_type_dict[type_name] = [rule]
        return val_type_dict

    def _collect_gen_identifier(self) -> dict[GenRuleId, GenRule]:

        rules = self.generation_rules
        genid_dict = {
            rule.identifier: rule for rule in rules
        }
        return genid_dict

    def _check_class(self, instances: list) -> GenRule:

        class_obj = instances[0].__class__
        for instance in instances[1:]:
            if instance.__class__ != class_obj:
                assert False, 'Rules should all belong to the same class!'
        return class_obj
