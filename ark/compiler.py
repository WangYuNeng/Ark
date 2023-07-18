"""Compiler for CDG to dynamical system simulation"""
import ast, inspect
from typing import Optional, Mapping, Any
from types import FunctionType
import copy
import numpy as np
import scipy
from ark.rewrite import RewriteGen
from ark.cdg.cdg import CDG, CDGNode, CDGEdge, CDGElement
from ark.specification.specification import CDGSpec
from ark.specification.production_rule import ProdRule, TIME, kw_name
from ark.reduction import Reduction


def ddt(name: str, order: int) -> str:
    """return the name of the derivative of the given order"""
    return f"{'ddt_' * order}{name}"

def rn_attr(name: str, attr: str) -> str:
    """return the variable name of the named attribute"""
    return f'{name}_{attr}'

def switch_attr(name: str) -> str:
    """return the name of the switch variable"""
    return rn_attr(name, 'switch')

def mk_var(name: str) -> ast.Name:
    """convert name to an ast.Name"""
    return ast.Name(name)

def mk_arg(name: str) -> ast.arg:
    """convert name to an ast.arg"""
    return ast.arg(arg=name)

def parse_expr(val: 'int | float | FunctionType | str') -> ast.expr:
    """parse value to expression"""
    if isinstance(val, int) or isinstance(val, float):
        val_str = str(val)
    elif isinstance(val, FunctionType):
        val_str = val.__name__
    elif isinstance(val, str):
        val_str = val
    else:
        raise TypeError(f'Unsupported type {type(val)}')
    mod = ast.parse(val_str)
    return mod.body[0]

def set_ctx(expr: ast.expr, ctx: ast.expr_context) -> ast.expr:
    """set the context of the given node and its children to the given context"""
    for nch in ast.iter_child_nodes(expr):
        if isinstance(nch, ast.expr):
            setattr(nch, 'ctx', ctx())
            set_ctx(nch, ctx)

    if isinstance(expr, ast.expr):
        setattr(expr, 'ctx', ctx())
    return expr

def concat_expr(exprs: ast.expr, operator: ast.BinOp) -> ast.BinOp:
    """concatenate expressions with the given operator"""
    if len(exprs) == 1:
        return exprs[0].body
    if len(exprs) == 2:
        return ast.BinOp(left=exprs[0].body, op=operator, right=exprs[1].body)
    return ast.BinOp(left=exprs[0].body, op=operator, right=concat_expr(exprs[1:], operator))

def apply_gen_rule(rule: ProdRule, transformer: RewriteGen) -> ast.expr:
    """Return the rewritten ast from the given rule"""
    gen_ast = copy.deepcopy(rule.fn_ast)
    transformer.visit(gen_ast)
    return gen_ast

class ArkCompiler():

    INIT_SEED, SIM_SEED = 'init_seed', 'sim_seed'
    INIT_RAND_EXPT = parse_expr(f'np.random.rand({INIT_SEED})')
    SIM_RAND_EXPT = parse_expr(f'np.random.rand({SIM_SEED})')
    ODE_FN_NAME, ODE_INPUT_VAR = '__ode_fn', '__variables'
    INIT_STATE, TIME_RANGE = 'init_states', 'time_range'
    SWITCH_VAL = 'switch_vals'
    KWARGS = 'kwargs'
    SIM_SOL = '__sol'
    SOLVE_IVP_EXPR = parse_expr(f'{SIM_SOL} = scipy.integrate.solve_ivp({ODE_FN_NAME}, \
                                {TIME_RANGE}, {INIT_STATE}, **{KWARGS})')
    PROG_NAME = 'dynamics'

    def __init__(self, rewrite: RewriteGen) -> None:
        self._rewrite = rewrite
        self._var_mapping = {}
        self._switch_mapping = {}
        self._namespace = {'np': np, 'scipy': scipy}
        self._prog_ast = None

    @property
    def prog_name(self):
        """name of the program to be generated"""
        return self.PROG_NAME

    @property
    def tmp_file_name(self):
        """name of the temporary file to store the generated program"""
        return f'__tmp_{self.prog_name}__.py'

    @property
    def var_mapping(self) -> dict[CDGNode, int]:
        """map CDGNode to the corresponding index in the state variables"""
        return self._var_mapping

    @property
    def switch_mapping(self) -> dict[CDGEdge, int]:
        """map CDGEdge to the corresponding for switch variables"""
        return self._switch_mapping

    def prog(self, time_range: tuple[float, float], init_states: list[float],
             switch_vals: Optional[list[bool]]=None,
             init_seed: Optional[int]=0, sim_seed: Optional[int]=0, **kwargs):
        """execute the compiled program
        
        kwargs: additional arguments for scipy.integrate.solve_ivp
        """
        return self._namespace[self.prog_name](time_range, init_states,
                                               switch_vals,
                                               init_seed, sim_seed, **kwargs)

    def print_prog(self):
        """print the compiled program"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._prog_ast))

    def map_init_state(self, node_to_val: Mapping[CDGNode, int | float]) -> list[int | float]:
        """map the initial state to the corresponding index in the state variables"""
        return self._mapping(node_to_val, self._var_mapping)

    def map_switch_val(self, edge_to_val: Mapping[CDGEdge, bool]) -> list[bool]:
        """map the switch values to the corresponding index in the switch variables"""
        return self._mapping(edge_to_val, self._switch_mapping)

    def _mapping(self, ele_to_val: Mapping[object, Any],
                 ele_to_idx: Mapping[object, int]) -> list[Any]:
        """map the values to the corresponding index"""
        assert ele_to_idx is not None, "Mapping is not generated yet!"
        assert len(ele_to_val) == len(ele_to_idx), "Elements mismatch!"
        vals = [None for _ in ele_to_idx]
        for ele, val in ele_to_val.items():
            vals[ele_to_idx[ele]] = val
        return vals

    def compile(self, cdg: CDG, cdg_spec: CDGSpec, help_fn: list[FunctionType], import_lib: dict):
        '''
        Compile the cdg to a function for dynamical system simulation
        help_fn: list of non-built-in function written in attributes, e.g., [sin, trapezoidal]
        import_lib: additional libraries, e.g., {'np': np}
        '''

        user_def_fns = self._compile_user_def_fn(help_fn)
        attr_var_def = self._compile_attribute_var(cdg)
        ode_fn = self._compile_ode_fn(cdg, cdg_spec)
        solv_ivp_stmts = self._compile_solve_ivp()

        stmts = user_def_fns + attr_var_def + [ode_fn] + solv_ivp_stmts
        top_stmt = self._compile_top_stmt(stmts)

        module = ast.Module([top_stmt], type_ignores=[])
        module = ast.fix_missing_locations(module)
        self._prog_ast = module
        code = compile(source=module, filename=self.tmp_file_name, mode='exec')
        self._namespace.update(import_lib)
        exec(code, self._namespace)

    def _compile_user_def_fn(self, funcs: list[FunctionType]) -> list[ast.FunctionDef]:
        """Compile user defined functions to ast.FunctionDef"""
        return [ast.parse(inspect.getsource(fn)).body[0] for fn in funcs]

    def _compile_attribute_var(self, cdg: CDG) -> list[ast.Assign]:
        """Compile the attributes of nodes and edges to variables"""

        ele: CDGElement

        stmts = [self.INIT_RAND_EXPT]
        if cdg.ds_order != 1:
            raise NotImplementedError('only support first order dynamical system now')
        self._var_mapping = {node: i for i, node in enumerate(cdg.nodes_in_order(1))}
        self._switch_mapping = {edge: i for i, edge in enumerate(cdg.switches)}

        # Map the input vector to the state variables
        if cdg.switches:
            stmts.append(ast.Assign([
                set_ctx(ast.Tuple([mk_var(switch_attr(edge.name)) for edge in cdg.switches]),
                        ast.Store)],
                set_ctx(mk_var(self.SWITCH_VAL), ast.Load)))

        for ele in cdg.nodes + cdg.edges:
            vname = ele.name
            lhs, rhs = [], []
            if ele.attrs.items():
                for attr_name in ele.attrs.keys():
                    lhs.append(mk_var(rn_attr(vname, attr_name)))
                    val_str = ele.get_attr_str(attr_name)
                    rhs.append(parse_expr(val_str).value)
                stmts.append(ast.Assign([
                    set_ctx(ast.Tuple(lhs), ast.Store)],
                    set_ctx(ast.Tuple(rhs), ast.Load)))
        return stmts

    def _compile_ode_fn(self, cdg: CDG, cdg_spec: CDGSpec) -> ast.FunctionDef:
        """Compile the cdg to the ode function for scipy.integrate.solve_ivp simulation"""

        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: ProdRule
        reduction: Reduction

        if cdg.ds_order != 1:
            raise NotImplementedError('only support first order dynamical system now')

        stmts = []
        input_vec = self.ODE_INPUT_VAR
        ode_fn_name = self.ODE_FN_NAME

        # Map the input vector to the state variables
        stmts.append(ast.Assign([
            set_ctx(ast.Tuple([mk_var(node.name) for node in cdg.nodes_in_order(1)]), ast.Store)],
            set_ctx(mk_var(input_vec), ast.Load)))

        # Generate the ddt_var = f(var) statements
        for order in range(cdg.ds_order + 1):
            for node in  cdg.nodes_in_order(order):
                vname = ddt(node.name, order=order)
                reduction = node.reduction
                rhs = []
                for edge in node.edges:
                    src, dst = edge.src, edge.dst
                    gen_rule = cdg_spec.match_gen_rule(edge=edge, src=src, dst=dst,
                                                    tgt=node.which_tgt(edge))
                    if gen_rule is not None:
                        self._rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                        rhs_expr = apply_gen_rule(rule=gen_rule, transformer=self._rewrite)
                        if edge.switchable:
                            body = ast.BinOp(left=rhs_expr.body, op=reduction.ast_switch,
                                             right=set_ctx(mk_var(switch_attr(edge.name)),
                                                           ast.Load))
                            rhs_expr = ast.Expression(body=body)
                        rhs.append(rhs_expr)
                if rhs:
                    stmts.append(ast.Assign(targets=[set_ctx(mk_var(vname), ast.Store)],
                                            value=concat_expr(rhs, reduction.ast_op)))

        # Return statement of the ode function
        stmts.append(set_ctx(ast.Return(ast.List([mk_var(ddt(node.name, order=1))
                                                  for node in cdg.nodes_in_order(1)])), ast.Load))

        arguments = ast.arguments(posonlyargs=[], args=[mk_arg(kw_name(TIME)), mk_arg(input_vec)],
                                 kwonlyargs=[], kw_defaults=[], defaults=[])
        return ast.FunctionDef(ode_fn_name, arguments, stmts, decorator_list=[])

    def _compile_solve_ivp(self) -> tuple[ast.Assign, ast.Return]:
        stmts = [self.SIM_RAND_EXPT, self.SOLVE_IVP_EXPR,
                 set_ctx(ast.Return(mk_var(self.SIM_SOL)), ast.Load)]
        return stmts

    def _compile_top_stmt(self, stmts) -> ast.FunctionDef:
        """Make a top level statement"""
        args = [
            mk_arg(self.TIME_RANGE),
            mk_arg(self.INIT_STATE),
            mk_arg(self.SWITCH_VAL),
            mk_arg(self.INIT_SEED),
            mk_arg(self.SIM_SEED),
        ]
        kwarg = mk_arg(self.KWARGS)
        defaults = [
            ast.Constant(None),
            ast.Constant(0),
            ast.Constant(0)
        ]
        return ast.FunctionDef(name=self.prog_name, args=ast.arguments(posonlyargs=[], args=args,
                                                                       kwonlyargs=[], kwarg=kwarg,
                                                                       kw_defaults=[],
                                                                       defaults=defaults),
                               body=stmts, decorator_list=[])
