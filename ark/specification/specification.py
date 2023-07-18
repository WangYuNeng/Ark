from ark.specification.cdg_types import NodeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.validation_rule import ValRule, ValRuleId

class CDGSpec:

    def __init__(self, cdg_types: list[NodeType], production_rules: list[ProdRule],
                 validation_rules: list[ValRule]):
        self._cdg_types = cdg_types
        self._production_rules = production_rules
        self._prod_rule_dict = self._collect_prod_identifier()

        if validation_rules is None:
            validation_rules = []
        self._validation_rules = validation_rules
        self._val_rule_dict = self._collect_type_to_val_rule()

    @property
    def cdg_types(self) -> list[NodeType]:
        """Access CDG types in the spec."""
        return self._cdg_types

    @property
    def production_rules(self) -> list[ProdRule]:
        """Access production rules in the spec."""
        return self._production_rules

    @property
    def prod_rule_dict(self) -> dict[ProdRuleId, ProdRule]:
        """Access mapping from production rule identifier to production rule."""
        return self._prod_rule_dict

    @property
    def validation_rules(self) -> list[ValRule]:
        """Access validation rules in the spec."""
        return self._validation_rules

    @property
    def val_rule_dict(self) -> dict[str, ValRule]:
        """Access mapping from validation rule identifier to validation rule."""
        return self._val_rule_dict

    def _collect_type_to_val_rule(self) -> dict[NodeType, ValRule]:
        return {val_rule.tgt_node_type: val_rule for val_rule in self.validation_rules}

    def _collect_prod_identifier(self) -> dict[ProdRuleId, ProdRule]:

        rules = self.production_rules
        genid_dict = {
            rule.identifier: rule for rule in rules
        }
        return genid_dict
