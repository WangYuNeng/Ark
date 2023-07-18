from dataclasses import dataclass
from typing import Callable, Optional
from ark.cdg.cdg import CDGNode
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.range import Range
from ark.specification.rule_keyword import Target, kw_name

@dataclass
class ValRuleId:
    """Validation Rule Identifier Class"""

    val_tgt: Target
    edge_type: EdgeType
    node_type: NodeType

    def __hash__(self) -> int:
        return repr(kw_name(self.val_tgt), self.edge_type.name, self.node_type.name).__hash__()

    def __str__(self) -> str:
        return str([kw_name(self.val_tgt), self.edge_type.name, self.node_type.name])

@dataclass
class ValPattern:
    """Pattern for a CDGNode.
    
    target: Which side of the edge that the node under validation is on.
    edge_type: The type of the edge
    node_types: list of acceptable node types
    deg_range: The range of acceptable degrees of this pattern.
    """

    def __init__(self, target: Target, edge_type: EdgeType,
                 node_types: list[NodeType], deg_range: Range) -> None:
        self._target = target
        self._edge_type = edge_type
        self._node_types = node_types
        self._deg_range = deg_range

    @property
    def edge_type(self):
        """The type of the edge."""
        return self._edge_type

    @property
    def target(self):
        """Which side of the edge that the node under validation is on."""
        return self._target

    @property
    def deg_range(self):
        """The range of acceptable degrees of this pattern."""
        return self._deg_range

    @property
    def node_types(self):
        """The list of acceptable node types."""
        return self._node_types

    @property
    def identifiers(self) -> set[ValRuleId]:
        """Set of uniqe identifiers for the subpatterns.
        
        Will expand all the node types into subpatterns and return a set of
        unique identifiers for them."""

        ids = set()
        for node_type in self.node_types:
            ids.add(self.get_identifier(self.target, self.edge_type, node_type))
        return ids

    @staticmethod
    def get_identifier(target, edge_type, node_type) -> ValRuleId:
        """Returns a unique identifier for the pattern"""
        return ValRuleId(target, edge_type, node_type)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kw_name(self.target)} \
            {self.edge_type} {self.node_types} {self.deg_range})'

class ValRule:
    """Validation rule for a CDGNode."""

    def __init__(self, tgt_node_type: NodeType,
                 ac_pats: Optional[list[ValPattern]]=None,
                 rej_pats: Optional[list[ValPattern]]=None,
                 checking_fns: Optional[list[Callable[[CDGNode], bool]]]=None
                 ) -> None:
        self._tgt_node_type = tgt_node_type
        if ac_pats is None:
            ac_pats = []
        self._ac_pats = ac_pats

        if rej_pats is None:
            rej_pats = []
        self._rej_pats = rej_pats

        if checking_fns is None:
            checking_fns = []
        self._checking_fns = checking_fns

    @property
    def tgt_node_type(self):
        """The target node type of this validation rule."""
        return self._tgt_node_type

    @property
    def ac_pats(self):
        """The accepted patterns of this validation rule."""
        return self._ac_pats

    @property
    def rej_pats(self):
        """The rejected patterns of this validation rule."""
        return self._rej_pats

    @property
    def checking_fns(self):
        """The custonchecking functions of this validation rule."""
        return self._checking_fns

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.tgt_node_type} \
            {self.ac_pats} {self.rej_pats} {self.checking_fns})'