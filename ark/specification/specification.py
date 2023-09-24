from typing import Optional
from ark.specification.cdg_types import CDGType, NodeType, EdgeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.validation_rule import ValRule


def access_inherit(method):
    def wrapper(self: "CDGSpec", inherit: bool):
        if inherit and self.inherit is not None:
            return method(
                self,
            ) + method(self.inh.b, inherit=True)
        else:
            return method(self)

    return wrapper


class CDGSpec:
    def __init__(
        self,
        name: str = "cdg-spec",
        cdg_types: Optional[list[CDGType]] = None,
        production_rules: Optional[list[ProdRule]] = None,
        validation_rules: Optional[list[ValRule]] = None,
        inherit: Optional["CDGSpec"] = None,
    ):
        self.name = name
        self.inherit = inherit
        if cdg_types is None:
            cdg_types = []
        if production_rules is None:
            production_rules = []
        if validation_rules is None:
            validation_rules = []

        self._node_types_dict, self._edge_types_dict = {}, {}
        self.add_cdg_types(cdg_types)
        self._prod_rule_dict = {}
        self.add_production_rules(production_rules)
        self._val_rule_dict = {}
        self.add_validation_rules(validation_rules)

    def node_type(self, name: str, inherit: bool = True) -> NodeType:
        """Access a node type in the spec."""
        node_types_dict = self.node_types_dict(inherit)
        if name not in node_types_dict:
            raise ValueError(f"Node type {name} not defined.")
        return node_types_dict[name]

    def edge_type(self, name: str, inherit: bool = True) -> EdgeType:
        """Access an edge type in the spec."""
        edge_types_dict = self.edge_types_dict(inherit)
        if name not in edge_types_dict:
            raise ValueError(f"Edge type {name} not defined.")
        return edge_types_dict[name]

    def cdg_types(self, inherit: bool = True) -> list[CDGType]:
        """Access CDG types in the spec."""
        return self.node_types(inherit) + self.edge_types(inherit)

    def node_types_dict(self, inherit: bool = True) -> dict[str, NodeType]:
        """Access mapping from node type name to node type."""
        return self._get_attr_inherit("_node_types_dict", inherit)

    def node_types(self, inherit: bool = True) -> list[NodeType]:
        """Access node types in the spec."""
        types_dict = self.node_types_dict(inherit)
        return list(types_dict.values())

    def nt_names(self, inherit: bool = True) -> list[str]:
        """Access node type names in the spec."""
        types_dict = self.node_types_dict(inherit)
        return list(types_dict.keys())

    def edge_types_dict(self, inherit: bool = True) -> dict[str, EdgeType]:
        """Access mapping from edge type name to edge type."""
        return self._get_attr_inherit("_edge_types_dict", inherit)

    def edge_types(self, inherit: bool = True) -> list[EdgeType]:
        """Access edge types in the spec."""
        types_dict = self.edge_types_dict(inherit)
        return list(types_dict.values())

    def et_names(self, inherit: bool = True) -> list[str]:
        """Access edge type names in the spec."""
        types_dict = self.edge_types_dict(inherit)
        return list(types_dict.values())

    def production_rules(self, inherit: bool = True) -> list[ProdRule]:
        """Access production rules in the spec."""
        return list(self.prod_rule_dict(inherit).values())

    def prod_rule_dict(self, inherit: bool = True) -> dict[ProdRuleId, ProdRule]:
        """Access mapping from production rule identifier to production rule."""
        return self._get_attr_inherit("_prod_rule_dict", inherit)

    def validation_rules(self, inherit: bool = True) -> list[ValRule]:
        """Access validation rules in the spec."""
        return list(self.val_rule_dict(inherit).values())

    def val_rule_dict(self, inherit: bool = True) -> dict[NodeType, ValRule]:
        """Access mapping from validation rule identifier to validation rule."""
        return self._get_attr_inherit("_val_rule_dict", inherit)

    def is_inherited(self, val: CDGType) -> bool:
        if self.inherit is None:
            return False
        if val in self.inherit.node_types():
            return True
        if val in self.inherit.edge_types():
            return True
        return False

    def _get_attr_inherit(self, name: str, inherit: bool) -> list | dict:
        """Access an attribute of the class.

        Args:
            name (str): the name of the attribute
            inherit (bool): access the attribute from the parent class or not
        """
        attr = getattr(self, name)
        if inherit and self.inherit is not None:
            attr_inherited = self.inherit._get_attr_inherit(name, inherit=True)
            if isinstance(attr, list) and isinstance(attr_inherited, list):
                attr_all = attr + attr_inherited
            elif isinstance(attr, dict) and isinstance(attr_inherited, dict):
                assert (
                    not attr.keys() & attr_inherited.keys()
                ), f"Duplicate keys in inherited attr {name}."
                attr_all = {**attr, **attr_inherited}
        else:
            attr_all = attr
        return attr_all

    def add_cdg_types(self, cdg_types: list[CDGType]) -> None:
        """Add a CDG type to the spec."""
        for cdg_type in cdg_types:
            if isinstance(cdg_type, NodeType):
                if cdg_type.name in self.node_types_dict():
                    raise ValueError("Node type already defined.")
                self._node_types_dict[cdg_type.name] = cdg_type
            elif isinstance(cdg_type, EdgeType):
                if cdg_type.name in self.edge_types_dict():
                    raise ValueError("Edge type already defined.")
                self._edge_types_dict[cdg_type.name] = cdg_type

    def add_production_rules(self, production_rules: list[ProdRule]) -> None:
        """Add a production rule to the spec."""
        new_dict = self.collect_prod_identifier(production_rules)
        if new_dict.keys() & self.prod_rule_dict().keys():
            raise ValueError("Production rule already defined.")
        self._prod_rule_dict.update(new_dict)

    def add_validation_rules(self, validation_rules: list[ValRule]) -> None:
        """Add a validation rule to the spec."""
        new_dict = self.collect_type_to_val_rule(validation_rules)
        if new_dict.keys() & self.val_rule_dict().keys():
            raise ValueError("Validation rule already defined.")
        self._val_rule_dict.update(new_dict)

    @staticmethod
    def collect_type_to_val_rule(val_rules: list[ValRule]) -> dict[NodeType, ValRule]:
        """Build a mapping from node type to validation rule."""
        return {val_rule.tgt_node_type: val_rule for val_rule in val_rules}

    @staticmethod
    def collect_prod_identifier(
        prod_rules: list[ProdRule],
    ) -> dict[ProdRuleId, ProdRule]:
        """Build a mapping from production rule identifier to production rule."""
        genid_dict = {rule.identifier: rule for rule in prod_rules}
        return genid_dict

    def reset_type_id(self) -> None:
        """Reset all id counters in cdg_type."""
        for cdg_type in self.cdg_types():
            cdg_type.reset_id()

    def filename(self, suffix: str, extension: str) -> str:
        """File name of the spec."""

        def format_filename(text):
            text = text.lower()
            return "-".join(text.split(" "))

        prefix = format_filename(self.name)
        suffix = format_filename(suffix)
        filename = "lang-%s-%s.%s" % (prefix, suffix, extension)
        return filename
