import ast

class RewriteGen(ast.NodeTransformer):

    def __init__(self) -> None:
        self._mapping = None
        super().__init__()

    @property
    def mapping(self):
        return self._mapping
    
    @mapping.setter
    def mapping(self, val):
        self._mapping = val
    
    def visit_Name(self, node: ast.Attribute):
        type_name, ctx = node.id, node.ctx
        return ast.Name(id=self.mapping[type_name], ctx=ctx)

    def visit_Attribute(self, node: ast.Attribute):
        value, attr, ctx = node.value, node.attr, node.ctx
        id = self.visit_Name(value).id
        new_name = '{}_{}'.format(id, attr)
        return ast.Name(id=new_name, ctx=ctx)