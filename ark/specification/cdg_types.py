'''
Meta classes for CDG types.
'''
from abc import abstractmethod
from typing import Optional, Mapping, Any
import inspect
from ark.reduction import Reduction, SUM
from ark.cdg.cdg import CDGElement, CDGNode, CDGEdge
from ark.specification.attribute_def import AttrDef, AttrImpl

def named_list_to_dict(attr_list: list[AttrDef]) -> dict[str, AttrDef]:
    """Convert a list of Attr to a dict of Attr"""
    return {attr.name: attr for attr in attr_list}

class CDGType(type):
    '''
    Base CDG type class.

    Keyword arguments:
    name -- name of this type
    parent_type -- parent type of this type
    attrs -- attributes of this type
    '''

    order: int
    reduction: Reduction
    attr_def: Mapping[str, AttrDef]

    def __init__(cls, **kwargs: Mapping[str, Any]):
        """
            Update the attributes of this class
            Otherwise, the attributes of the parent class will be used and the 
            derived class will not have correct attributes
            TODO: check if this is the correct way to do this.
                Might have some reference issues with update.
        """
        attr_def = named_list_to_dict(kwargs.get('attr_def', {}))
        cls.attr_def.update(attr_def)
        cls.new_instance_id = 0
        super().__init__(cls)

    def __call__(cls, **attrs: Mapping[str, AttrImpl]) -> CDGElement:
        if cls.check_attr(**attrs):
            element_name = cls.new_name() # Why does this trigger a pylint error?
            return super().__call__(cdg_type=cls, name=element_name, **attrs)

    def check_attr(cls, **attrs: Mapping[str, AttrImpl]) -> bool:
        """Check whether the given attributes are valid for this CDGType"""
        attr_def = cls.attr_def
        if attr_def.keys() != attrs.keys():
            raise AttributeError(f'{attr_def.keys()} has different attributes than {attrs.keys()}')
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
        return f'{type_name}_{id}'

    @property
    def name(cls) -> str:
        """
        The user-defined name of the CDGType
        """
        return cls.__name__

class NodeType(CDGType):
    """CDG node type.

    Keyword arguments:
    name -- name of this node type
    base -- parent NodeType of this node type
    attrs -- attributes of this node type
    order -- the derivative taken in the dynamical system of this node type
    """
    def __new__(mcs, name: str, base: Optional[CDGType]=None, attr_def: Optional[list[AttrDef]]=None,
                order: Optional[int]=0, reduction: Optional[Reduction]=SUM):

        if attr_def is None:
            attr_def = []
        attr_def = named_list_to_dict(attr_def)

        if base is None:
            base = CDGNode
        else:
            order = base.order
            reduction = base.reduction
            # Probably due to the way __getattribute__ works setting attrs here does not work
            # Update in the __init__ method looks like a workaround.
            # Still put this here for now in case of future changes.
            attr_def = base.attr_def.copy()
            attr_def.update(attr_def)

        class_attrs = {'order': order, 'reduction': reduction, 'attr_def': attr_def}
        bases = (base,)
        return super().__new__(mcs, name, bases, class_attrs)

    def base_cdg_types(cls) -> 'list[NodeType]':
        base_types = filter(lambda x: isinstance(x, NodeType), inspect.getmro(cls))
        return list(base_types)

class EdgeType(CDGType):
    """CDG edge type.

    Keyword arguments:
    name -- name of this node type
    parent_type -- parent NodeType of this node type
    attrs -- attributes of this node type
    """
    def __new__(mcs, name: str, base: Optional[CDGType]=None, attr_def: Optional[list[AttrDef]]=None):
        if base is None:
            bases = (CDGEdge,)
        attr_def = named_list_to_dict(attr_def)
        class_attrs = {'attr_def': attr_def}
        return super().__new__(mcs, name, bases, class_attrs)

    def base_cdg_types(cls) -> "list[EdgeType]":
        base_types = filter(lambda x: isinstance(x, EdgeType), inspect.getmro(cls))
        return list(base_types)
