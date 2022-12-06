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

    SRC, DST = False, True
    SRC_STR, DST_STR = 'SRC', 'DST'

    def __init__(self, tgt_et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: bool, fn_exp: str, transformer=RewriteGen()) -> None:
        self._tgt_et = tgt_et
        self._src_nt = src_nt
        self._dst_nt = dst_nt
        self._gen_tgt = gen_tgt
        self._fn_ast = ast.parse(fn_exp, mode='eval')
        self._transformer = transformer

    def hook_transformer(self, transformer: RewriteGen):
        self._transformer = transformer

    @staticmethod
    def get_identifier(tgt_et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: bool):
        gen_tgt = bool(gen_tgt)
        return repr([tgt_et.type_name, src_nt.type_name, dst_nt.type_name, gen_tgt])

    @property
    def identifier(self):
        return self.get_identifier(self._tgt_et, self._src_nt, self._dst_nt, self._gen_tgt)

    def apply(self, edge: CDGEdge, src_node: CDGNode, dst_node: CDGNode):
        name_map = {self._tgt_et.type_name: edge.name, self.SRC_STR: src_node.name, self.DST_STR: dst_node.name}
        self._transformer.name_map = name_map
        gen_ast = copy.deepcopy(self._fn_ast)
        self._transformer.visit(gen_ast)
        return gen_ast
    
if __name__ == '__main__':

    gen_rule_exp = '1/E.qd*SRC+DST.c'

    et = EdgeType('E', {'qs': [], 'qd': []})
    nt = StatefulNodeType('VN', {})

    es, ns = [], []
    for i in range(4):
        es.append(CDGEdge(i, f'e{i}', et, []))
    
    for i in range(5):
        ns.append(CDGNode(i, f'n{i}', nt, []))

    rule = GenRule(tgt_et=et, src_nt=nt, dst_nt=nt, gen_tgt=0, fn_exp=gen_rule_exp, transformer=RewriteGen())

    for i in range(4):
        gen_ast = rule.apply(edge=es[i], src_node=ns[i], dst_node=ns[i+1])
        print(ast.unparse(gen_ast))

    print(isinstance(nt, StatefulNodeType))