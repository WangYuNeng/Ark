import ast, inspect
from types import FunctionType
import copy

from ark.rewrite import RewriteGen
from ark.cdg.cdg import CDG, CDGNode, CDGEdge, CDGElement
from ark.specification.specification import CDGSpec
from ark.specification.generation_rule import GenRule, TIME, kw_name
from ark.reduction import Reduction

class ArkCompiler():

    def __init__(self, rewrite: RewriteGen) -> None:
        self._rewrite = rewrite
        self._var_mapping = {}
        self._namespace = {}

    @property
    def prog_name(self):
        """name of the program to be generated"""
        return 'dynamics'

    @property
    def var_mapping(self) -> dict:
        """map variable (node.name) to the corresponding index in the state variables"""
        return self._var_mapping

    def prog(self):
        """return the compiled program"""
        return self._namespace[self.prog_name]

    def compile(self, cdg: CDG, cdg_spec: CDGSpec, help_fn: list, import_lib: list):
        '''
        Compile the cdg to a function for dynamical system simulation
        help_fn: list of non-built-in function written in attributes, e.g., [sin, trapezoidal]
        import_lib: additional libraries, e.g., {'np': np}
        '''

        def ddt(name: str, order: int) -> str:
            return f"{'ddt_' * order}{name}"

        def rn_attr(name: str, attr: str) -> str:
            return f'{name}_{attr}'

        def mk_var(name: str) -> ast.Name:
            return ast.Name(name)

        def mk_arg(name: str) -> ast.arg:
            return ast.arg(arg=name)

        def parse_expr(val: 'int | float | FunctionType | str') -> ast.Expr:
            if isinstance(val, int) or isinstance(val, float):
                val = str(val)
            elif isinstance(val, FunctionType):
                val = val.__name__
            mod = ast.parse(val)
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
            if len(exprs) == 1:
                return exprs[0].body
            if len(exprs) == 2:
                return ast.BinOp(left=exprs[0].body, op=op(), right=exprs[1].body)
            return ast.BinOp(left=exprs[0].body, op=op(), right=concat_expr(exprs[1:], op))

        ele: CDGElement
        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: GenRule
        reduction: Reduction

        stmts = []

        for fn in help_fn:
            stmts.append(ast.parse(inspect.getsource(fn)).body[0])

        input_vec = '__VARIABLES'
        if cdg.ds_order != 1:
            raise NotImplementedError('only support first order dynamical system now')

        stmts.append(ast.Assign([
            set_ctx(ast.Tuple([mk_var(node.name) for node in cdg.nodes_in_order(1)]), ast.Store)],
            set_ctx(mk_var(input_vec), ast.Load)))
        for ele in cdg.nodes + cdg.edges:
            vname = ele.name
            lhs, rhs = [], []
            if ele.attrs.items():
                for attr, val in ele.attrs.items():
                    lhs.append(mk_var(rn_attr(vname, attr)))
                    rhs.append(parse_expr(val))
                stmts.append(ast.Assign([
                    set_ctx(ast.Tuple(lhs), ast.Store)],
                    set_ctx(ast.Tuple(rhs), ast.Load)))

        for order in range(cdg.ds_order + 1):
            for node in  cdg.nodes_in_order(order):
                vname = ddt(node.name, order=order)
                reduction = node.reduction
                rhs = []
                for edge in node.edges:
                    src, dst = edge.src, edge.dst
                    gen_rule = cdg_spec.match_gen_rule(edge=edge, src=src, dst=dst,
                                                    tgt=node.gen_tgt_type(edge))
                    if gen_rule is not None:
                        self._rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                        rhs.append(self._apply_rule(edge=edge, rule=gen_rule,
                                                    transformer=self._rewrite))
                if rhs:
                    stmts.append(ast.Assign(targets=[set_ctx(mk_var(vname), ast.Store)],
                                            value=concat_expr(rhs, reduction.ast_op())))

        self._var_mapping = {node: i for i, node in enumerate(cdg.nodes_in_order(1))}
        stmts.append(set_ctx(ast.Return(ast.List([mk_var(ddt(node.name, order=1))
                                                  for node in cdg.nodes_in_order(1)])), ast.Load))

        arguments =ast.arguments(posonlyargs=[], args=[mk_arg(kw_name(TIME)), mk_arg(input_vec)], kwonlyargs=[], kw_defaults=[], defaults=[])
        top_stmts = []
        top_stmts.append(ast.FunctionDef(self.prog_name, arguments, stmts, decorator_list=[]))

        module = ast.Module(top_stmts, type_ignores=[])
        module = ast.fix_missing_locations(module)
        code = compile(source=module, filename='__tmp_{}.py'.format(self.prog_name), mode='exec')
        self._namespace = import_lib
        print(ast.unparse(module))
        exec(code, self._namespace)

    def _apply_rule(self, edge: CDGEdge, rule: GenRule, transformer: RewriteGen):
        gen_ast = copy.deepcopy(rule.fn_ast)
        transformer.visit(gen_ast)
        return gen_ast