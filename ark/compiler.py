"""Compiler for CDG to dynamical system simulation"""
import ast
import inspect
from typing import Optional, Mapping, Any
from types import FunctionType
import copy
from itertools import product
import numpy as np
import sympy
from tqdm import tqdm
import scipy
from ark.rewrite import BaseRewriteGen
from ark.cdg.cdg import CDG, CDGNode, CDGEdge, CDGElement
from ark.specification.specification import CDGSpec
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.rule_keyword import Target, TIME, kw_name
from ark.reduction import Reduction


def ddt(name: str, order: int) -> str:
    """return the name of the derivative of the given order"""
    return f"{'ddt_' * order}{name}"


def rn_attr(name: str, attr: str) -> str:
    """return the variable name of the named attribute"""
    return f"{name}_{attr}"


def switch_attr(name: str) -> str:
    """return the name of the switch variable"""
    return rn_attr(name, "switch")


def mk_var(name: str, to_sympy: bool = False) -> ast.Name:
    """convert name to an ast.Name"""
    if to_sympy:
        return sympy.Symbol(name)
    return ast.Name(name)


def mk_arg(name: str) -> ast.arg:
    """convert name to an ast.arg"""
    return ast.arg(arg=name)


def parse_expr(val: "int | float | FunctionType | str") -> ast.expr:
    """parse value to expression"""
    if isinstance(val, int) or isinstance(val, float):
        val_str = str(val)
    elif isinstance(val, FunctionType):
        val_str = val.__name__
    elif isinstance(val, str):
        val_str = val
    else:
        raise TypeError(f"Unsupported type {type(val)}")
    mod = ast.parse(val_str)
    return mod.body[0]


def set_ctx(expr: ast.expr, ctx: ast.expr_context) -> ast.expr:
    """set the context of the given node and its children to the given context"""
    for nch in ast.iter_child_nodes(expr):
        if isinstance(nch, ast.expr):
            setattr(nch, "ctx", ctx())
            set_ctx(nch, ctx)

    if isinstance(expr, ast.expr):
        setattr(expr, "ctx", ctx())
    return expr


def concat_expr(
    exprs: list[ast.expr] | list[sympy.Expr], operator: ast.operator | type
) -> ast.operator | sympy.Expr:
    """concatenate expressions with the given operator"""
    if isinstance(operator, ast.operator):
        if len(exprs) == 1:
            return exprs[0].body
        if len(exprs) == 2:
            return ast.BinOp(left=exprs[0].body, op=operator, right=exprs[1].body)
        return ast.BinOp(
            left=exprs[0].body, op=operator, right=concat_expr(exprs[1:], operator)
        )
    else:
        return operator(*exprs)


def apply_gen_rule(
    rule: ProdRule, transformer: BaseRewriteGen, to_sympy: bool = False
) -> ast.expr | sympy.Expr:
    """Return the rewritten ast from the given rule"""
    if not to_sympy:
        gen_expr = copy.deepcopy(rule.fn_ast)
    else:
        gen_expr = rule.fn_sympy
    rr_expr = transformer.visit(gen_expr)
    return rr_expr


def match_prod_rule(
    rule_dict: dict[ProdRuleId, ProdRule],
    edge: CDGEdge,
    src: CDGNode,
    dst: CDGNode,
    tgt: Target,
) -> ProdRule:
    """
    Find the production rule that matches the given edge, source node,
    destination node, and rule target.

    TODO: Handle multiple matches.
    """

    def check_match(
        edge_type: EdgeType, src_nt: NodeType, dst_nt: NodeType
    ) -> ProdRule | None:
        rule_id = ProdRuleId(edge_type, src_nt, dst_nt, tgt)
        if rule_id in rule_dict:
            return rule_dict[rule_id]
        return None

    edge_type: EdgeType = edge.cdg_type
    src_nt: NodeType = src.cdg_type
    dst_nt: NodeType = dst.cdg_type

    match = check_match(edge_type, src_nt, dst_nt)
    if match is not None:
        return match

    et_base = edge_type.base_cdg_types()
    src_nt_base = src_nt.base_cdg_types()
    dst_nt_base = dst_nt.base_cdg_types()

    for edge_type, src_nt, dst_nt in product(et_base, src_nt_base, dst_nt_base):
        match = check_match(edge_type, src_nt, dst_nt)
        if match is not None:
            if (
                edge_type == et_base[-1]
                and src_nt == src_nt_base[-1]
                and dst_nt == dst_nt_base[-1]
            ):
                return match
            raise NotImplementedError("Have not implemented match in the heirarchy.")

    return None


class ArkCompiler:
    """Ark compiler for CDG to dynamical system simulation

    Args:
    RewriteGen: ast rewrite class
    """

    INIT_SEED, SIM_SEED = "init_seed", "sim_seed"
    INIT_RAND_EXPT = parse_expr(f"np.random.rand({INIT_SEED})")
    SIM_RAND_EXPT = parse_expr(f"np.random.rand({SIM_SEED})")
    ODE_FN_NAME, ODE_INPUT_VAR = "__ode_fn", "__variables"
    INIT_STATE, TIME_RANGE = "init_states", "time_range"
    SWITCH_VAL = "switch_vals"
    KWARGS = "kwargs"
    SIM_SOL = "__sol"
    SOLVE_IVP_EXPR = parse_expr(
        f"{SIM_SOL} = scipy.integrate.solve_ivp({ODE_FN_NAME}, \
                                {TIME_RANGE}, {INIT_STATE}, **{KWARGS})"
    )
    PROG_NAME = "dynamics"

    def __init__(self, rewrite: BaseRewriteGen) -> None:
        self._rewrite = rewrite
        self._var_mapping = {}
        self._switch_mapping = {}
        self._namespace = {"np": np, "scipy": scipy}
        self._prog_ast = None
        self._gen_rule_dict = {}
        self._verbose = 0

    @property
    def prog_name(self):
        """name of the program to be generated"""
        return self.PROG_NAME

    @property
    def tmp_file_name(self):
        """name of the temporary file to store the generated program"""
        return f"__tmp_{self.prog_name}__.py"

    @property
    def var_mapping(self) -> dict[CDGNode, int]:
        """map CDGNode to the corresponding index in the state variables"""
        return self._var_mapping

    @property
    def switch_mapping(self) -> dict[CDGEdge, int]:
        """map CDGEdge to the corresponding for switch variables"""
        return self._switch_mapping

    def prog(
        self,
        time_range: tuple[float, float],
        init_states: list[float],
        switch_vals: Optional[list[bool]] = None,
        init_seed: Optional[int] = 0,
        sim_seed: Optional[int] = 0,
        **kwargs,
    ):
        """execute the compiled program

        kwargs: additional arguments for scipy.integrate.solve_ivp
        """
        return self._namespace[self.prog_name](
            time_range, init_states, switch_vals, init_seed, sim_seed, **kwargs
        )

    def print_prog(self):
        """print the compiled program"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._prog_ast))

    def dump_prog(self, file_name: str):
        """dump the compiled program to the given file"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        with open(file_name, "w") as f:
            f.write(ast.unparse(self._prog_ast))

    def map_init_state(
        self, node_to_val: Mapping[CDGNode, int | float]
    ) -> list[int | float]:
        """map the initial state to the corresponding index in the state variables"""
        return self._mapping(node_to_val, self._var_mapping)

    def map_switch_val(self, edge_to_val: Mapping[CDGEdge, bool]) -> list[bool]:
        """map the switch values to the corresponding index in the switch variables"""
        return self._mapping(edge_to_val, self._switch_mapping)

    def _mapping(
        self, ele_to_val: Mapping[object, Any], ele_to_idx: Mapping[object, int]
    ) -> list[Any]:
        """map the values to the corresponding index"""
        assert ele_to_idx is not None, "Mapping is not generated yet!"
        assert len(ele_to_val) == len(ele_to_idx), "Elements mismatch!"
        vals = [None for _ in ele_to_idx]
        for ele, val in ele_to_val.items():
            vals[ele_to_idx[ele]] = val
        return vals

    def compile(
        self,
        cdg: CDG,
        cdg_spec: CDGSpec,
        help_fn: list[FunctionType],
        import_lib: dict,
        verbose: int = 0,
        inline_attr: bool = False,
    ):
        """
        Compile the cdg to a function for dynamical system simulation
        help_fn: list of non-built-in function written in attributes, e.g., [sin, trapezoidal]
        import_lib: additional libraries, e.g., {'np': np}
        verbose: 0: no verbose, 1: print the compilation progress
        inline_attr: inline the attribute definitions into dynamical system function.
                    Note that this will cause bug if the attributes are random variable.
        """

        self._rewrite.set_attr_rn_fn(rn_attr)
        self._verbose = verbose
        user_def_fns = self._compile_user_def_fn(help_fn)
        attr_var_def, attr_mapping = self._compile_attribute_var(
            cdg, inline=inline_attr
        )
        self._rewrite.attr_mapping = attr_mapping
        ode_fn = self._compile_ode_fn(cdg, cdg_spec)
        solv_ivp_stmts = self._compile_solve_ivp()

        stmts = user_def_fns + attr_var_def + [ode_fn] + solv_ivp_stmts
        top_stmt = self._compile_top_stmt(stmts)

        module = ast.Module([top_stmt], type_ignores=[])
        module = ast.fix_missing_locations(module)
        self._prog_ast = module

        if verbose:
            print("Compiling the program...")
        code = compile(source=module, filename=self.tmp_file_name, mode="exec")
        self._namespace.update(import_lib)
        exec(code, self._namespace)

        if verbose:
            print("Compilation finished")

    def compile_sympy(
        self,
        cdg: CDG,
        cdg_spec: CDGSpec,
        help_fn: list[FunctionType],
    ) -> list[sympy.Expr]:
        """Compile a CDG to sympy expressions

        Args:
            cdg (CDG): The input CDG
            cdg_spec (CDGSpec): Specification
            help_fn (list[FunctionType]): List of non-built-in functions
        """
        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: ProdRule
        reduction: Reduction

        rule_dict = cdg_spec.prod_rule_dict()
        if cdg.ds_order != 1:
            raise NotImplementedError("only support first order dynamical system now")

        stmts = []

        # Generate the ddt_var = f(var) statements
        for order in range(cdg.ds_order + 1):
            nodes = tqdm(
                cdg.nodes_in_order(order), desc=f"Compiling order {order} nodes"
            )

            for node in nodes:
                vname = ddt(node.name, order=order)
                reduction = node.reduction
                rhs = []
                for edge in node.edges:
                    src, dst = edge.src, edge.dst
                    gen_rule = match_prod_rule(
                        rule_dict=rule_dict,
                        edge=edge,
                        src=src,
                        dst=dst,
                        tgt=node.which_tgt(edge),
                    )
                    if gen_rule is not None:
                        self._rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                        rhs_expr = apply_gen_rule(
                            rule=gen_rule, transformer=self._rewrite, to_sympy=True
                        )
                        if edge.switchable:
                            raise NotImplementedError
                        rhs.append(rhs_expr)
                if rhs:
                    stmts.append(
                        (
                            mk_var(vname, to_sympy=True),
                            concat_expr(rhs, reduction.sympy_op),
                        )
                    )
        return stmts

    def _compile_user_def_fn(self, funcs: list[FunctionType]) -> list[ast.FunctionDef]:
        """Compile user defined functions to ast.FunctionDef"""
        return [ast.parse(inspect.getsource(fn)).body[0] for fn in funcs]

    def _compile_attribute_var(self, cdg: CDG, inline: bool) -> list[ast.Assign]:
        """Compile the attributes of nodes and edges to variables"""

        ele: CDGElement

        stmts = [self.INIT_RAND_EXPT]
        if cdg.ds_order != 1:
            raise NotImplementedError("only support first order dynamical system now")
        self._var_mapping = {node: i for i, node in enumerate(cdg.nodes_in_order(1))}
        self._switch_mapping = {edge: i for i, edge in enumerate(cdg.switches)}

        # Map the input vector to the state variables
        if cdg.switches:
            stmts.append(
                ast.Assign(
                    [
                        set_ctx(
                            ast.Tuple(
                                [
                                    mk_var(switch_attr(edge.name))
                                    for edge in cdg.switches
                                ]
                            ),
                            ast.Store,
                        )
                    ],
                    set_ctx(mk_var(self.SWITCH_VAL), ast.Load),
                )
            )

        attr_to_ast_node = {}
        if self._verbose:
            eles = tqdm(cdg.nodes + cdg.edges, desc="Compiling attributes")
        else:
            eles = cdg.nodes + cdg.edges

        for ele in eles:
            vname = ele.name
            lhs, rhs = [], []
            if ele.attrs.items():
                for attr_name in ele.attrs.keys():
                    renamed_attr = rn_attr(vname, attr_name)
                    attr_expr_str = ele.get_attr_str(attr_name)
                    attr_expr = parse_expr(attr_expr_str).value
                    if inline:
                        attr_to_ast_node[renamed_attr] = attr_expr
                    else:
                        attr_var = mk_var(renamed_attr)
                        lhs.append(attr_var)
                        rhs.append(attr_expr)
                        attr_to_ast_node[renamed_attr] = mk_var(renamed_attr)
                if lhs and rhs:
                    stmts.append(
                        ast.Assign(
                            [set_ctx(ast.Tuple(lhs), ast.Store)],
                            set_ctx(ast.Tuple(rhs), ast.Load),
                        )
                    )
        return stmts, attr_to_ast_node

    def _compile_ode_fn(self, cdg: CDG, cdg_spec: CDGSpec) -> ast.FunctionDef:
        """Compile the cdg to the ode function for scipy.integrate.solve_ivp simulation"""

        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: ProdRule
        reduction: Reduction

        rule_dict = cdg_spec.prod_rule_dict()
        if cdg.ds_order != 1:
            raise NotImplementedError("only support first order dynamical system now")

        stmts = []
        input_vec = self.ODE_INPUT_VAR
        ode_fn_name = self.ODE_FN_NAME

        # Map the input vector to the state variables
        stmts.append(
            ast.Assign(
                [
                    set_ctx(
                        ast.Tuple(
                            [mk_var(node.name) for node in cdg.nodes_in_order(1)]
                        ),
                        ast.Store,
                    )
                ],
                set_ctx(mk_var(input_vec), ast.Load),
            )
        )

        # Generate the ddt_var = f(var) statements
        for order in range(cdg.ds_order + 1):
            if self._verbose:
                nodes = tqdm(
                    cdg.nodes_in_order(order), desc=f"Compiling order {order} nodes"
                )
            else:
                nodes = cdg.nodes_in_order(order)

            for node in nodes:
                vname = ddt(node.name, order=order)
                reduction = node.reduction
                rhs = []
                for edge in node.edges:
                    src, dst = edge.src, edge.dst
                    gen_rule = match_prod_rule(
                        rule_dict=rule_dict,
                        edge=edge,
                        src=src,
                        dst=dst,
                        tgt=node.which_tgt(edge),
                    )
                    if gen_rule is not None:
                        self._rewrite.name_mapping = gen_rule.get_rewrite_mapping(
                            edge=edge
                        )
                        rhs_expr = apply_gen_rule(
                            rule=gen_rule, transformer=self._rewrite
                        )
                        if edge.switchable:
                            body = ast.BinOp(
                                left=rhs_expr.body,
                                op=reduction.ast_switch,
                                right=set_ctx(mk_var(switch_attr(edge.name)), ast.Load),
                            )
                            rhs_expr = ast.Expression(body=body)
                        rhs.append(rhs_expr)
                if rhs:
                    stmts.append(
                        ast.Assign(
                            targets=[set_ctx(mk_var(vname), ast.Store)],
                            value=concat_expr(rhs, reduction.ast_op),
                        )
                    )

        # Return statement of the ode function
        stmts.append(
            set_ctx(
                ast.Return(
                    ast.List(
                        [
                            mk_var(ddt(node.name, order=1))
                            for node in cdg.nodes_in_order(1)
                        ]
                    )
                ),
                ast.Load,
            )
        )

        arguments = ast.arguments(
            posonlyargs=[],
            args=[mk_arg(kw_name(TIME)), mk_arg(input_vec)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        return ast.FunctionDef(ode_fn_name, arguments, stmts, decorator_list=[])

    def _compile_solve_ivp(self) -> tuple[ast.Assign, ast.Return]:
        stmts = [
            self.SIM_RAND_EXPT,
            self.SOLVE_IVP_EXPR,
            set_ctx(ast.Return(mk_var(self.SIM_SOL)), ast.Load),
        ]
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
        defaults = [ast.Constant(None), ast.Constant(0), ast.Constant(0)]
        return ast.FunctionDef(
            name=self.prog_name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                kwonlyargs=[],
                kwarg=kwarg,
                kw_defaults=[],
                defaults=defaults,
            ),
            body=stmts,
            decorator_list=[],
        )
