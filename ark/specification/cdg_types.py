"""
Meta classes for CDG types.
"""
import inspect
from abc import abstractmethod
from typing import Any, Mapping, Optional

from ark.cdg.cdg import CDGEdge, CDGElement, CDGNode
from ark.reduction import SUM, Reduction
from ark.specification.attribute_def import AttrDef, AttrImpl


def named_list_to_dict(attr_list: list[AttrDef]) -> dict[str, AttrDef]:
    """Convert a list of Attr to a dict of Attr"""
    return {attr.name: attr for attr in attr_list}


class CDGType(type):
    """
    Base CDG type class.

    Args:
        name: Name of this node type.
        bases: Parent Type of this node type.
        attrs: Attributes of this node type, e.g., "order", "reduction", "attr_def"
    """

    order: int
    reduction: Reduction
    attr_def: Mapping[str, AttrDef]

    def __init__(
        cls,
        name: str,
        bases: Optional[list] = None,
        attrs: Optional[dict[str, Any]] = None,
    ):
        """
        Update the attributes of this class
        Otherwise, the attributes of the parent class will be used and the
        derived class will not have correct attributes
        TODO: check if this is the correct way to do this.
            Might have some reference issues with update.
        """
        if attrs is None:
            attr_def = {}
        else:
            attr_def = attrs["attr_def"]
        cls.attr_def.update(attr_def)
        cls.new_instance_id = 0
        super().__init__(cls)

    def __call__(cls, **attrs: Mapping[str, AttrImpl]) -> CDGElement:
        if cls.check_attr(**attrs):
            element_name = cls.new_name()  # Why does this trigger a pylint error?
            # Somehow this calls the CDGNode/CDGEdge __init__ method as specified in
            # the bases respectively and I forgot why I know this works.
            return super().__call__(cdg_type=cls, name=element_name, **attrs)

    def check_attr(cls, **attrs: Mapping[str, AttrImpl]) -> bool:
        """Check whether the given attributes are valid for this CDGType"""
        attr_def = cls.attr_def
        if attr_def.keys() != attrs.keys():
            raise AttributeError(
                f"{attr_def.keys()} has different attributes than {attrs.keys()}"
            )
        for attr, value in attrs.items():
            if not attr_def[attr].check(value):
                return False
        return True

    def new_name(cls) -> str:
        """Create a unique name for a new instance of this CDGType"""
        element_name = cls.instance_name(cls.name, cls.new_instance_id)
        cls.new_instance_id += 1
        return element_name

    @abstractmethod
    def base_cdg_types(cls) -> "list[CDGType]":
        """Return the base CDG types of this CDGType"""
        raise NotImplementedError

    @staticmethod
    def instance_name(type_name: str, id: int) -> str:
        """Unique name of an instance of this CDGType"""
        return f"{type_name}_{id}"

    @property
    def name(cls) -> str:
        """
        The user-defined name of the CDGType
        """
        return cls.__name__

    def reset_id(cls) -> None:
        """
        Reset the instance id counter
        """
        cls.new_instance_id = 0


class NodeType(CDGType):
    """CDG node type.

    Args:
        name: Name of this node type.
        bases: Parent NodeType of this node type.
        attrs: Attributes of this node type, e.g., "order", "reduction", "attr_def"
    """

    def __new__(
        mcs,
        name: str,
        bases: Optional[CDGType] = (CDGNode,),
        attrs: Optional[dict[str, Any]] = None,
    ):
        if not attrs or "attr_def" not in attrs:
            attr_def = {}
        else:
            attr_def = attrs["attr_def"]

        # Case0: Inherit another NodeType
        if isinstance(bases, NodeType) or (
            len(bases) == 1 and isinstance(bases[0], NodeType)
        ):
            if isinstance(bases, NodeType):
                bases = (bases,)
            base = bases[0]
            if "order" in attrs:
                order = attrs["order"]
                assert (
                    order == base.order
                ), f"Inherited type orrder ({order}) should be the same with\
                    the base type order ({base.order})"
            if "reduction" in attrs:
                reduction = attrs["reduction"]
                assert (
                    reduction == base.reduction
                ), f"Inherited type reduction ({reduction}) should be the same with\
                    the base type reduction ({base.reduction})"
            order = base.order
            reduction = base.reduction
            attr_def = base.attr_def.copy()
            attr_def.update(attr_def)

        # Case1: Default, take CDGNode as base
        elif bases[0] == CDGNode:
            if "order" not in attrs:
                raise ValueError("order should be specified when base is not specified")
            order = attrs["order"]

            if "reduction" not in attrs:
                reduction = SUM
            else:
                reduction = attrs["reduction"]

        else:
            raise ValueError(
                "Unrecognized base classes \
                (only support single inheritance for NodeType)"
            )

        class_attrs = {"order": order, "reduction": reduction, "attr_def": attr_def}
        return super().__new__(mcs, name, bases, class_attrs)

    def __call__(cls, **attrs: Mapping[str, AttrImpl]) -> CDGNode:
        node: CDGNode = super().__call__(**attrs)
        return node

    def base_cdg_types(cls) -> "list[NodeType]":
        base_types = filter(lambda x: isinstance(x, NodeType), inspect.getmro(cls))
        return list(base_types)


class EdgeType(CDGType):
    """CDG edge type.

    Args:
        name: Name of this edge type.
        bases: Parent EdgeType of this edge type.
        attrs: Attributes of this node type, e.g., "attr_def"
    """

    def __new__(
        mcs,
        name: str,
        bases: Optional[CDGType] = (CDGEdge,),
        attrs: Optional[dict[str, AttrDef]] = None,
    ):
        if not attrs or "attr_def" not in attrs:
            attr_def = {}
        else:
            attr_def = attrs["attr_def"]

        # Case0: Inherit another EdgeType
        if isinstance(bases, EdgeType) or (
            len(bases) == 1 and isinstance(bases[0], EdgeType)
        ):
            if isinstance(bases, EdgeType):
                bases = (bases,)
            base = bases[0]
            attr_def = base.attr_def.copy()
            attr_def.update(attr_def)

        # Case1: Default, take CDGEdge as base
        elif bases[0] == CDGEdge:
            pass

        else:
            raise ValueError(
                "Unrecognized base classes \
                (only support single inheritance for EdgeType)"
            )

        class_attrs = {"attr_def": attr_def}
        return super().__new__(mcs, name, bases, class_attrs)

    def __call__(cls, switchable: bool = False, **attrs: Mapping[str, Any]) -> CDGEdge:
        edge: CDGEdge = super().__call__(**attrs)
        edge.switchable = switchable
        return edge

    def base_cdg_types(cls) -> "list[EdgeType]":
        base_types = filter(lambda x: isinstance(x, EdgeType), inspect.getmro(cls))
        return list(base_types)
