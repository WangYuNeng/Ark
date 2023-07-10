'''
Meta classes for CDG types.
'''
from typing import Dict
from ark.reduction import Reduction, SUM, PRODUCT
from ark.cdg.cdg import CDGElement, CDGNode, CDGEdge

class CDGType(type):
    '''
    Base CDG type class.

    Keyword arguments:
    name -- name of this type
    parent_type -- parent type of this type
    attrs -- attributes of this type
    '''

    def __init__(cls, *args, **kwargs):
        """
            Update the attributes of this class
            Otherwise, the attributes of the parent class will be used and the 
            derived class will not have correct attributes
            TODO: check if this is the correct way to do this.
                Might have some reference issues with update.
        """
        cls.attr_defs.update(kwargs.get('attr_defs', {}))
        super().__init__(cls)

    def __call__(cls, **attrs) -> CDGElement:
        cls._check_attr(**attrs)
        return super().__call__(cls, **attrs)

    def _check_attr(cls, **attrs) -> bool:
        attr_defs = cls.attr_defs
        if attr_defs.keys() != attrs.keys():
            raise AttributeError(f'{attr_defs.keys()} has different attributes than {attrs.keys()}')
        for attr, value in attrs.items():
            if not attr_defs[attr].check(value):
                return False
        return True

class NodeType(CDGType):
    """CDG node type.

    Keyword arguments:
    name -- name of this node type
    parent_type -- parent NodeType of this node type
    attrs -- attributes of this node type
    order -- the derivative taken in the dynamical system of this node type
    """
    def __new__(mcs, name: str, inherit: CDGType=None, attr_defs: Dict=None,
                order: int=0, reduction: Reduction=SUM):

        if inherit is None:
            inherit = CDGNode
        else:
            order = inherit.order
            reduction = inherit.reduction
            # Probably due to the way __getattribute__ works setting attr_defs here does not work
            # Update in the __init__ method looks like a workaround.
            # Still put this here for now in case of future changes.
            attr_defs = inherit.attr_defs.copy()
            attr_defs.update(attr_defs)

        class_attrs = {'order': order, 'reduction': reduction, 'attr_defs': attr_defs}
        bases = (inherit,)
        return super().__new__(mcs, name, bases, class_attrs)

class EdgeType(CDGType):
    """CDG edge type.

    Keyword arguments:
    name -- name of this node type
    parent_type -- parent NodeType of this node type
    attrs -- attributes of this node type
    """
    def __new__(mcs, name: str, inherit=None, attr_defs=None):
        if inherit is None:
            bases = (CDGEdge,)
        class_attrs = {'attr_defs': attr_defs}
        return super().__new__(mcs, name, bases, class_attrs)
