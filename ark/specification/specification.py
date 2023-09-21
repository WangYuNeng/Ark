from typing import Optional
from ark.specification.cdg_types import CDGType, NodeType, EdgeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.validation_rule import ValRule

class CDGSpec:

    def __init__(self, cdg_types: Optional[list[CDGType]]=None,
                 production_rules: Optional[list[ProdRule]]=None,
                 validation_rules: Optional[list[ValRule]]=None):

        if cdg_types is None:
            cdg_types = []
        if production_rules is None:
            production_rules = []
        if validation_rules is None:
            validation_rules = []

        self._node_types_dict, self._edge_types_dict= {}, {}
        self.add_cdg_types(cdg_types)
        self._production_rules, self._prod_rule_dict = [], {}
        self.add_production_rules(production_rules)
        self._validation_rules, self._val_rule_dict = [], {}
        self.add_validation_rules(validation_rules)

    def node_type(self, name: str) -> NodeType:
        """Access a node type in the spec."""
        if name not in self._node_types_dict:
            raise ValueError(f'Node type {name} not defined.')
        return self._node_types_dict[name]
    
    def edge_type(self, name: str) -> EdgeType:
        """Access an edge type in the spec."""
        if name not in self._edge_types_dict:
            raise ValueError(f'Edge type {name} not defined.')
        return self._edge_types_dict[name]

    @property
    def cdg_types(self) -> list[CDGType]:
        """Access CDG types in the spec."""
        return self.node_types + self.edge_types
    
    @property
    def node_types(self) -> list[NodeType]:
        """Access node types in the spec."""
        return list(self._node_types_dict.values())
    
    @property
    def nt_names(self) -> list[str]:
        """Access node type names in the spec."""
        return self._node_types_dict.keys()
    
    @property
    def edge_types(self) -> list[EdgeType]:
        """Access edge types in the spec."""
        return list(self._edge_types_dict.values())
    
    @property
    def et_names(self) -> list[str]:
        """Access edge type names in the spec."""
        return self._edge_types_dict.keys()

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
    def val_rule_dict(self) -> dict[NodeType, ValRule]:
        """Access mapping from validation rule identifier to validation rule."""
        return self._val_rule_dict

    def add_cdg_types(self, cdg_types: list[CDGType]) -> None:
        """Add a CDG type to the spec."""
        for cdg_type in cdg_types:
            if isinstance(cdg_type, NodeType):
                if cdg_type.name in self._node_types_dict:
                    raise ValueError('Node type already defined.')
                self._node_types_dict[cdg_type.name] = cdg_type
            elif isinstance(cdg_type, EdgeType):
                if cdg_type.name in self._edge_types_dict:
                    raise ValueError('Edge type already defined.')
                self._edge_types_dict[cdg_type.name] = cdg_type

    def add_production_rules(self, production_rules: list[ProdRule]) -> None:
        """Add a production rule to the spec."""
        self._production_rules.extend(production_rules)
        new_dict = self.collect_prod_identifier(production_rules)
        if new_dict.keys() & self._prod_rule_dict.keys():
            raise ValueError('Production rule already defined.')
        self._prod_rule_dict.update(new_dict)

    def add_validation_rules(self, validation_rules: list[ValRule]) -> None:
        """Add a validation rule to the spec."""
        self._validation_rules.extend(validation_rules)
        new_dict = self.collect_type_to_val_rule(validation_rules)
        if new_dict.keys() & self._val_rule_dict.keys():
            raise ValueError('Validation rule already defined.')
        self._val_rule_dict.update(new_dict)

    @staticmethod
    def collect_type_to_val_rule(val_rules: list[ValRule]) -> dict[NodeType,
                                                                   ValRule]:
        """Build a mapping from node type to validation rule."""
        return {val_rule.tgt_node_type: val_rule for val_rule in val_rules}

    @staticmethod
    def collect_prod_identifier(prod_rules: list[ProdRule]) -> dict[ProdRuleId,
                                                                    ProdRule]:
        """Build a mapping from production rule identifier to production rule."""
        genid_dict = {
            rule.identifier: rule for rule in prod_rules
        }
        return genid_dict

    def reset_type_id(self) -> None:
        """Reset all id counters in cdg_type."""
        for cdg_type in self.cdg_types:
            cdg_type.reset_id()
