'''
Meta classes for CDG types.
'''
from typing import Dict
from ark.reduction import Reduction, SUM
from ark.cdg.cdg import CDGElement, CDGNode, CDGEdge

class CDGType(type):
    '''
    Base CDG type class.

    Keyword arguments:
    name -- name of this type
    parent_type -- parent type of this type
    attrs -- attributes of this type
    '''

    def __init__(cls, **kwargs):
        """
            Update the attributes of this class
            Otherwise, the attributes of the parent class will be used and the 
            derived class will not have correct attributes
            TODO: check if this is the correct way to do this.
                Might have some reference issues with update.
        """
        cls.attr_defs.update(kwargs.get('attr_defs', {}))
        cls.new_instance_id = 0
        super().__init__(cls)

    def __call__(cls, **attrs) -> CDGElement:
        print(cls.__dict__)
        element_name = cls.new_name() # Why does this trigger a pylint error?
        return super().__call__(cdg_type=cls, name=element_name, **attrs)

    def check_attr(cls, **attrs) -> bool:
        """Check whether the given attributes are valid for this CDGType"""
        attr_defs = cls.attr_defs
        if attr_defs.keys() != attrs.keys():
            raise AttributeError(f'{attr_defs.keys()} has different attributes than {attrs.keys()}')
        for attr, value in attrs.items():
            if not attr_defs[attr].check(value):
                return False
        return True

    def new_name(cls) -> str:
        """Create a unique name for a new instance of this CDGType"""
        element_name = cls.instance_name(cls.name, cls.new_instance_id)
        cls.new_instance_id += 1
        return element_name

    @staticmethod
    def instance_name(type_name, id):
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
    parent_type -- parent NodeType of this node type
    attrs -- attributes of this node type
    order -- the derivative taken in the dynamical system of this node type
    """
    def __new__(mcs, name: str, base: CDGType=None, attr_defs: Dict=None,
                order: int=0, reduction: Reduction=SUM):

        if base is None:
            base = CDGNode
        else:
            order = base.order
            reduction = base.reduction
            # Probably due to the way __getattribute__ works setting attr_defs here does not work
            # Update in the __init__ method looks like a workaround.
            # Still put this here for now in case of future changes.
            attr_defs = base.attr_defs.copy()
            attr_defs.update(attr_defs)

        class_attrs = {'order': order, 'reduction': reduction, 'attr_defs': attr_defs}
        bases = (base,)
        return super().__new__(mcs, name, bases, class_attrs)

class EdgeType(CDGType):
    """CDG edge type.

    Keyword arguments:
    name -- name of this node type
    parent_type -- parent NodeType of this node type
    attrs -- attributes of this node type
    """
    def __new__(mcs, name: str, base: CDGType=None, attr_defs: Dict=None):
        if base is None:
            bases = (CDGEdge,)
        class_attrs = {'attr_defs': attr_defs}
        return super().__new__(mcs, name, bases, class_attrs)
