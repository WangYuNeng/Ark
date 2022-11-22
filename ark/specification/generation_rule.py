import ast
import copy
from ark.specification.types import EdgeType, NodeType, StatefulNodeType
from ark.cdg .cdg import CDGEdge, CDGNode

class RewriteGen(ast.NodeTransformer):

    def __init__(self) -> None:
        self._name_map = None
        super().__init__()

    @property
    def name_map(self):
        return self._name_map
    
    @name_map.setter
    def name_map(self, val):
        self._name_map = val
    
    def visit_Name(self, node: ast.Attribute):
        type_name, ctx = node.id, node.ctx
        return ast.Name(id=self.name_map[type_name], ctx=ctx)

class GenRule:

    SRC, DST = 'SRC', 'DST'

    def __init__(self, tgt_et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: bool, fn_ast: ast.Expr, transformer: ast.NodeTransformer) -> None:
        self.tgt_et = tgt_et
        self.src_nt = src_nt
        self.dst_nt = dst_nt
        self.gen_tgt = gen_tgt
        self.fn_ast = fn_ast
        self.transformer = transformer

    def get_identifier(self):
        return [self.tgt_et.name, self.src_nt.name, self.dst_nt.name, self.gen_tgt]

    def apply(self, edge: CDGEdge, src_node: CDGNode, dst_node: CDGNode):
        name_map = {self.tgt_et.type_name: edge.name, self.SRC: src_node.name, self.DST: dst_node.name}
        self.transformer.name_map = name_map
        gen_ast = copy.deepcopy(self.fn_ast)
        self.transformer.visit(gen_ast)
        return gen_ast
    
if __name__ == '__main__':

    gen_rule_exp = '1/E.qd*SRC+DST.c'

    es, ns = [], []
    for i in range(4):
        es.append(CDGEdge(i, f'e{i}'))
    
    for i in range(5):
        ns.append(CDGNode(i, f'n{i}'))

    et = EdgeType('E', ['qs', 'qd'], [])
    nt = StatefulNodeType('VN', [], [])
    fn_ast = ast.parse(gen_rule_exp, mode='eval')

    rule = GenRule(tgt_et=et, src_nt=nt, dst_nt=nt, gen_tgt=0, fn_ast=fn_ast, transformer=RewriteGen())

    for i in range(4):
        gen_ast = rule.apply(edge=es[i], src_node=ns[i], dst_node=ns[i+1])
        print(ast.unparse(gen_ast))