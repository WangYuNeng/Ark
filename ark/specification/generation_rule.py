import ast
import copy
from ark.specification.cdg_types import EdgeType, NodeType
from ark.cdg .cdg import CDGEdge, CDGNode

class GenRule:

    SRC, DST = False, True
    SRC_STR, DST_STR = 'SRC', 'DST'

    def __init__(self, tgt_et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: bool, fn_exp: str) -> None:
        self._tgt_et = tgt_et
        self._src_nt = src_nt
        self._dst_nt = dst_nt
        self._gen_tgt = gen_tgt
        self._fn_ast = ast.parse(fn_exp, mode='eval')

    @staticmethod
    def get_identifier(tgt_et: EdgeType, src_nt: NodeType, dst_nt: NodeType, gen_tgt: bool):
        gen_tgt = bool(gen_tgt)
        return repr([tgt_et.type_name, src_nt.type_name, dst_nt.type_name, gen_tgt])

    @property
    def identifier(self):
        return self.get_identifier(self._tgt_et, self._src_nt, self._dst_nt, self._gen_tgt)

    @property
    def fn_ast(self):
        return self._fn_ast

    def get_rewrite_mapping(self, edge: CDGEdge):
        src, dst = edge.src, edge.dst
        name_map = {self._tgt_et.type_name: edge.name, self.SRC_STR: src.name, self.DST_STR: dst.name}
        return name_map
