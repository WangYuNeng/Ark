"""
Ast rewrite classes to generate expression from production rules
"""
import ast
from typing import Callable

class RewriteGen(ast.NodeTransformer):
    """
    Base rewrite class
    - name_mapping: dict[str, str], map name to another name
    - attr_mapping: dict[str, ast.expr], map attribute to an ast expression
    - attr_rn_fn: Callable[[str, str], str], attribute renaming function
    """

    def __init__(self) -> None:
        self._name_mapping = None
        self._attr_mapping = None
        self._attr_rn_fn = None
        super().__init__()

    @property
    def name_mapping(self) -> dict[str, str]:
        """mapping between names"""
        return self._name_mapping

    @name_mapping.setter
    def mapping(self, val: dict[str, str]) -> None:
        self._name_mapping = val

    @property
    def attr_mapping(self) -> dict[str, ast.expr]:
        """map attribute string to an ast node"""
        return self._attr_mapping

    @attr_mapping.setter
    def attr_mapping(self, val: dict[str, ast.expr]) -> None:
        self._attr_mapping = val

    def set_attr_rn_fn(self, val: Callable[[str, str], str]) -> None:
        """set attribute renaming function"""
        self._attr_rn_fn = val

    def visit_Name(self, node: ast.Name):
        """rewrite the name in the ast with name_mapping"""
        type_name, ctx = node.id, node.ctx
        return ast.Name(id=self.mapping[type_name], ctx=ctx)

    def visit_Attribute(self, node: ast.Attribute):
        """rewrite the attribute in the ast with attr_mapping"""
        value, attr, ctx = node.value, node.attr, node.ctx
        id = self.visit_Name(value).id
        new_name = self._attr_rn_fn(id, attr)
        ast_node = self.attr_mapping[new_name]
        ast_node.ctx = ctx
        return ast_node
