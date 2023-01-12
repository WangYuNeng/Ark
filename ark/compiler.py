import ast, inspect
import copy

from ark.rewrite import RewriteGen
from ark.cdg.cdg import CDG, CDGNode, CDGEdge, CDGElement
from ark.specification.specification import CDGSpec
from ark.specification.generation_rule import GenRule

class ArkCompiler():

    def __init__(self, rewrite: RewriteGen) -> None:
        self._rewrite = rewrite
        self._namespace = {}
        pass

    @property
    def prog_name(self):
        return 'dynamics'

    def prog(self):
        return self._namespace[self.prog_name]

    def compile(self, cdg: CDG, cdg_spec: CDGSpec, help_fn):

        def ddt(v):
            return 'ddt_{}'.format(v)

        def rn_attr(v, attr):
            return '{}_{}'.format(v, attr)

        def mk_var(v):
            return ast.Name(v)

        def mk_arg(v):
            return ast.arg(arg=v)

        def parse_expr(text):
            mod = ast.parse(text)
            return mod.body[0].value

        def set_ctx(n, ctx):
            for nch in ast.iter_child_nodes(n):
                if isinstance(nch, ast.expr):
                    setattr(nch, 'ctx', ctx())
                    set_ctx(nch, ctx)

            if isinstance(n, ast.expr):
                setattr(n, 'ctx', ctx())
            return n

        def concat_expr(exprs, op):
            if len(exprs) == 2:
                return ast.BinOp(left=exprs[0].body, op=op(), right=exprs[1].body)
            return ast.BinOp(left=exprs[0].body, op=op(), right=concat_expr(exprs[1:], op).body)

        ele: CDGElement
        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: GenRule

        stmts = []

        for fn in help_fn:
            stmts.append(ast.parse(inspect.getsource(fn)).body[0])
        
        input_vec = '__VARIABLES'
        stmts.append(ast.Assign([
            set_ctx(ast.Tuple([mk_var(node.name) for node in cdg.stateful_nodes]), ast.Store)],
            set_ctx(mk_var(input_vec), ast.Load)))
        for ele in cdg.nodes + cdg.edges:
            vname = ele.name
            lhs, rhs = [], []
            for attr, val in ele.attrs.items():
                lhs.append(mk_var(rn_attr(vname, attr)))
                rhs.append(parse_expr(val))
            stmts.append(ast.Assign([
                set_ctx(ast.Tuple(lhs), ast.Store)],
                set_ctx(ast.Tuple(rhs), ast.Load)))
            


        rule_dict = cdg_spec.gen_rule_dict
        rule_class = cdg_spec.gen_rule_class
        for node in cdg.stateful_nodes:
            # print(node)
            vname = ddt(node.name)
            rhs = []
            for edge in node.edges:
                src, dst = edge.src, edge.dst
                id = rule_class.get_identifier(tgt_et=edge.cdg_type, src_nt=src.cdg_type, dst_nt=dst.cdg_type, gen_tgt=node.is_dst(edge))
                gen_rule = rule_dict[id]
                self._rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                rhs.append(self._apply_rule(edge=edge, rule=rule_dict[id], transformer=self._rewrite))
            stmts.append(ast.Assign(targets=[set_ctx(mk_var(vname), ast.Store)], value=concat_expr(rhs, ast.Add)))
        
        stmts.append(set_ctx(ast.Return(ast.List([mk_var(ddt(node.name)) for node in cdg.stateful_nodes])), ast.Load))

        arguments =ast.arguments(posonlyargs=[], args=[mk_arg('t'), mk_arg(input_vec)], kwonlyargs=[], kw_defaults=[], defaults=[])
        top_stmts = []
        top_stmts.append(ast.FunctionDef(self.prog_name, arguments, stmts, decorator_list=[]))

        module = ast.Module(top_stmts, type_ignores=[])
        module = ast.fix_missing_locations(module)
        code = compile(source=module, filename='__tmp_{}.py'.format(self.prog_name), mode='exec')
        self._namespace = {}
        print(ast.unparse(module))
        exec(code, self._namespace)

    def _apply_rule(self, edge: CDGEdge, rule: GenRule, transformer: RewriteGen):
        gen_ast = copy.deepcopy(rule.fn_ast)
        transformer.visit(gen_ast)
        return gen_ast