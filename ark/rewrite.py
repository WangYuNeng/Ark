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