"""Compiler for CDG to dynamical system simulation"""

import ast
import copy
import inspect
from functools import partial
from itertools import product
from types import FunctionType

import jax.numpy as jnp
import numpy as np
import scipy
import sympy
from scipy import integrate
from tqdm import tqdm

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.reduction import Reduction
from ark.rewrite import BaseRewriteGen, RewriteGen
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.rule_keyword import SRC, TIME, Target, kw_name
from ark.specification.specification import CDGSpec


def ddt(name: str, order: int) -> str:
    """return the name of the derivative of the given order"""
    return f"{'ddt_' * order}{name}"


def n_state(name: str) -> str:
    """return the name of the next state"""
    return f"next_{name}"


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


def mk_list_assign(targets: list[ast.Name], value: ast.expr) -> ast.Assign:
    return ast.Assign(
        [
            set_ctx(
                ast.Tuple(targets),
                ast.Store,
            )
        ],
        set_ctx(value, ast.Load),
    )


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
) -> ProdRule | None:
    """Find the production rule that matches the given edge, source node,
    destination node, and rule target.

    Args:
        rule_dict (dict[ProdRuleId, ProdRule]): production rule dictionary
        edge (CDGEdge): edge to match
        src (CDGNode): source node to match
        dst (CDGNode): destination node to match
        tgt (Target): target of the production rule (SRC or DST)
    Returns:
        ProdRule: the matched production rule or None if no match is found
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

    # Try to match the node type of the target as close as possible
    # then the edge type and then the node type of the other end.
    if tgt == SRC:
        ordered_types = product(src_nt_base, et_base, dst_nt_base)
    else:
        ordered_types = product(dst_nt_base, et_base, src_nt_base)

    for nt0, et, nt1 in ordered_types:
        if tgt == SRC:
            src_nt, dst_nt, edge_type = nt0, nt1, et
        else:
            src_nt, dst_nt, edge_type = nt1, nt0, et
        match = check_match(edge_type, src_nt, dst_nt)
        if match is not None:
            return match

    return None


class ArkCompiler:
    """Ark compiler for CDG to dynamical system simulation

    Args:
    RewriteGen: ast rewrite class
    """

    SIM_SEED = "sim_seed"
    SIM_RAND_EXPT = parse_expr(f"np.random.rand({SIM_SEED})")
    ODE_FN_NAME, ODE_INPUT_VAR = "__ode_fn", "__variables"
    INIT_STATE, TIME_RANGE = "init_states", "time_range"
    SWITCH_VAL = "switch_vals"
    ATTR_VAL = "attr_vals"
    ARGS = "args"
    FN_ARGS = "fargs"
    KWARGS = "kwargs"
    SIM_SOL = "__sol"
    SOLVE_IVP_EXPR = parse_expr(
        f"{SIM_SOL} = scipy.integrate.solve_ivp({ODE_FN_NAME}, \
                                {TIME_RANGE}, {INIT_STATE}, **{KWARGS})"
    )
    PROG_NAME = "dynamics"

    def __init__(self, rewrite: BaseRewriteGen = RewriteGen()) -> None:
        self._rewrite = rewrite
        self._node_to_state_var = {}
        self._ode_fn_io_names = []
        self._switch_mapping = {}
        self._namespace = {
            "np": np,
            "scipy": scipy,
            "scipy.integrate": integrate,
            "jnp": jnp,
        }
        self._prog_ast = None
        self._ode_term_ast = None
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

    def print_prog(self):
        """print the compiled program"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._prog_ast))

    def dump_prog(self, file_name: str):
        """dump the compiled program to the given file"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        with open(file_name, "w") as f:
            f.write(ast.unparse(self._prog_ast))

    def print_odeterm(self):
        """print the compiled odeterm function"""
        assert self._ode_term_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._ode_term_ast))

    def dump_odeterm(self, file_name: str):
        """dump the compiled odeterm function to the given file"""
        assert self._ode_term_ast is not None, "Program is not compiled yet!"
        with open(file_name, "w") as f:
            f.write(ast.unparse(self._ode_term_ast))

    def compile(self, cdg: CDG, cdg_spec: CDGSpec, verbose: int = 0):
        """Compile the cdg to a function for dynamical system simulation.

        Args:
            cdg (CDG): The input graph to compile
            cdg_spec (CDGSpec): Specification containing the production rules
            verbose (int): level of status printing, 0 -- no printing,
            1 -- print the compilation progress.

        Returns:
            prog (FunctionType): The compiled function.
            node_mapping (dict[str, int]): map the name of CDGNode to the corresponding
            index in the state variables. The node_mapping[name] points to the index
            of the 0th order term and the n-th order deravitves (if applicable) index
            is node_mapping[name] + n.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the attribute variables. The attr_mapping[name][attr]
            points to the index of the attribute.
        """

        self._rewrite.set_attr_rn_fn(rn_attr)
        self._verbose = verbose

        (
            node_mapping,
            ode_fn_io_names,
            switch_mapping,
            attr_mapping,
        ) = self._collect_ode_fn_io(cdg)

        attr_var_def = self._compile_attribute_var(switch_mapping, attr_mapping)
        ode_fn, _ = self._compile_ode_fn(cdg, cdg_spec, ode_fn_io_names)
        solv_ivp_stmts = self._compile_solve_ivp()

        stmts = attr_var_def + [ode_fn] + solv_ivp_stmts
        top_stmt = self._compile_top_stmt(stmts)

        module = ast.Module([top_stmt], type_ignores=[])
        module = ast.fix_missing_locations(module)
        self._prog_ast = module

        if verbose:
            print("Compiling the program...")
        code = compile(source=module, filename=self.tmp_file_name, mode="exec")
        exec(code, self._namespace)

        if verbose:
            print("Compilation finished")

        return (
            self._namespace[self.prog_name],
            node_mapping,
            switch_mapping,
            attr_mapping,
        )

    def compile_odeterm(self, cdg: CDG, cdg_spec: CDGSpec, verbose: int = 0):
        """Compile the cdg to an executable function representing the derivative term.

        The function will in the form of `__ode_fn(time, __variables, args, fargs)`
        where `args` is the list of arguments mapped to
        the float/int attribute values and switch values, and `fargs` is the list of
        function arguments mapped to the FunctionType attribute values.

        The separation of the two arguments is for compatibility to Diffrax
        1. The function arguments can't be jitted. Need to be treat as static value
        separately.
        2. The ode term in Diffrax is in the form of f(t, y, args).

        Args:
            cdg (CDG): The input graph to compile
            cdg_spec (CDGSpec): Specification containing the production rules
            verbose (int): level of status printing, 0 -- no printing,
            1 -- print the compilation progress.

        Returns:
            prog (FunctionType): The compiled function.
            node_mapping (dict[str, int]): map the name of CDGNode to the corresponding
            index in the state variables. The node_mapping[name] points to the index
            of the 0th order term and the n-th order deravitves (if applicable) index
            is node_mapping[name] + n.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the attribute variables. The attr_mapping[name][attr]
            points to the index of the attribute.
        """

        self._rewrite.set_attr_rn_fn(rn_attr)
        self._verbose = verbose

        stmts = []

        (
            node_mapping,
            ode_fn_io_names,
            switch_mapping,
            attr_mapping,
        ) = self._collect_ode_fn_io(cdg, separate_fn_attr=True)

        # Count the numerical value attributes index from the switch mapping
        # Isolate the function attributes mapping
        num_attr_mapping, num_attr_idx = {}, len(switch_mapping)
        fn_attr_mapping, fn_attr_idx = {}, 0
        for ele_name, attrs in attr_mapping.items():
            num_attr_mapping[ele_name], fn_attr_mapping[ele_name] = {}, {}
            for attr_name, is_num in attrs.items():
                if not is_num:
                    fn_attr_mapping[ele_name][attr_name] = fn_attr_idx
                    fn_attr_idx += 1
                else:
                    num_attr_mapping[ele_name][attr_name] = num_attr_idx
                    num_attr_idx += 1

        # Generate the variable names of the switches and attributes
        args_names = [
            None
            for _ in range(
                len(switch_mapping)
                + sum(len(attrs) for attrs in num_attr_mapping.values())
            )
        ]
        fn_args_names = [
            None for _ in range(sum(len(attrs) for attrs in fn_attr_mapping.values()))
        ]

        for name, idx in switch_mapping.items():
            args_names[idx] = mk_var(switch_attr(name))
        for ele_name, attrs in num_attr_mapping.items():
            for attr_name, idx in attrs.items():
                args_names[idx] = mk_var(rn_attr(ele_name, attr_name))
        for ele_name, attrs in fn_attr_mapping.items():
            for attr_name, idx in attrs.items():
                fn_args_names[idx] = mk_var(rn_attr(ele_name, attr_name))

        # Generate the Assignment statements
        if args_names:
            stmts.append(
                mk_list_assign(
                    targets=args_names,
                    value=mk_var(self.ARGS),
                )
            )
        if fn_args_names:
            stmts.append(
                mk_list_assign(
                    targets=fn_args_names,
                    value=mk_var(self.FN_ARGS),
                )
            )
        _, ode_stmts = self._compile_ode_fn(
            cdg, cdg_spec, ode_fn_io_names, return_jax=True
        )
        stmts.extend(ode_stmts)
        arguments = ast.arguments(
            posonlyargs=[],
            args=[
                mk_arg(kw_name(TIME)),
                mk_arg(self.ODE_INPUT_VAR),
                mk_arg(self.ARGS),
                mk_arg(self.FN_ARGS),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        func_def = ast.FunctionDef(
            name=self.ODE_FN_NAME, args=arguments, body=stmts, decorator_list=[]
        )
        module = ast.Module([func_def], type_ignores=[])
        module = ast.fix_missing_locations(module)
        self._ode_term_ast = module

        code = compile(source=module, filename=self.tmp_file_name, mode="exec")
        exec(code, self._namespace)

        return (
            self._namespace[self.ODE_FN_NAME],
            node_mapping,
            switch_mapping,
            num_attr_mapping,
            fn_attr_mapping,
        )

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
        raise NotImplementedError

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

    def _compile_attribute_var(
        self,
        switch_mapping: dict[str, int],
        attr_mapping: dict[str, dict[str, int]],
    ) -> list[ast.Assign]:
        """Compile the attributes of nodes and edges to variables"""

        stmts = []

        # Map the input vector to the state variables
        if switch_mapping:
            edge_names = [None for _ in range(len(switch_mapping))]
            for name, idx in switch_mapping.items():
                edge_names[idx] = name
            stmts.append(
                mk_list_assign(
                    targets=[
                        mk_var(switch_attr(edge_name)) for edge_name in edge_names
                    ],
                    value=mk_var(self.SWITCH_VAL),
                )
            )

        if attr_mapping:
            attr_names = [
                None for _ in range(sum(len(attrs) for attrs in attr_mapping.values()))
            ]
            for ele_name, attrs in attr_mapping.items():
                for attr_name, idx in attrs.items():
                    attr_names[idx] = rn_attr(ele_name, attr_name)
            stmts.append(
                mk_list_assign(
                    targets=[mk_var(attr_name) for attr_name in attr_names],
                    value=mk_var(self.ATTR_VAL),
                )
            )
        return stmts

    def _collect_ode_fn_io(
        self, cdg: CDG, separate_fn_attr: bool = False
    ) -> tuple[dict[str, int], list[str], dict[str, int], dict[str, dict[str, int]]]:
        """Collect the input/output node mapping and var names of the ode function

        Args:
            cdg (CDG): The input CDG
            separate_fn_attr (bool): Separate the int/float attributes with function
            attributes.

        Returns:
            node_to_state_var (dict[str, int]): map name ofCDGNode to the corresponding
            index in the state variables. The `node_to_state_var[name]` points to the
            index of the 0th order term and the n-th order deravitves (if applicable)
            index is `node_to_state_var[name] + n`.
            ode_input_return_names (list): names of the state variables of the compiled
            ode function for scipy.integrate.solve_ivp simulation.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the attribute variables. The `attr_mapping[name][attr]`
            points to the index of the attribute if `separate_fn_attr` is false. Otherwise,
            it denote whether the attribute is a float/int attribute or not.
        """

        # Go through the nodes to collect the state variables
        # n-th order variable has n-1 state variables
        node_to_state_var = {}
        ode_input_return_names = []
        for node_order in range(1, cdg.ds_order + 1):
            for node in cdg.nodes_in_order(node_order):
                node_to_state_var[node.name] = len(ode_input_return_names)
                for order in range(node_order):
                    ode_input_return_names.append(ddt(node.name, order=order))

        # Go through the edges to collect the switch variables
        switch_mapping = {edge.name: i for i, edge in enumerate(cdg.switches)}
        attr_mapping, attr_idx = {}, 0
        for ele in cdg.nodes + cdg.edges:
            attr_mapping[ele.name] = {}
            for attr_name, attr in ele.attrs.items():
                if separate_fn_attr:
                    if isinstance(attr, FunctionType) or isinstance(attr, partial):
                        attr_mapping[ele.name][attr_name] = False
                    else:
                        attr_mapping[ele.name][attr_name] = True
                else:
                    attr_mapping[ele.name][attr_name] = attr_idx
                    attr_idx += 1

        return (
            node_to_state_var,
            ode_input_return_names,
            switch_mapping,
            attr_mapping,
        )

    def _compile_ode_fn(
        self, cdg: CDG, cdg_spec: CDGSpec, io_names: list[str], return_jax: bool = False
    ) -> tuple[ast.FunctionDef, list[ast.stmt]]:
        """Compile the cdg to the ode function for scipy.integrate.solve_ivp simulation

        The compiled function will have the following signature:
        ```
        def __ode_fn(t, y):
            var1, var2, ... = y
            return [ddt_var1, ddt_var2, ...]
        ```

        Args:
            cdg (CDG): The input CDG
            cdg_spec (CDGSpec): Specification
            io_names (list[str]): names of the state variables `var1, var2, ...`
            return_jax (bool): whether the return value is a list or a jax array.

        """

        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: ProdRule
        reduction: Reduction

        rule_dict = cdg_spec.prod_rule_dict()

        stmts = []
        input_vec = self.ODE_INPUT_VAR
        ode_fn_name = self.ODE_FN_NAME
        input_return_names = io_names

        stmts.append(
            mk_list_assign(
                targets=[mk_var(name) for name in input_return_names],
                value=set_ctx(mk_var(input_vec), ast.Load),
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

            # 0th order: var = f(vars)
            # 1th order: ddt_var = f(vars)
            # 2th order: ddt_var = ddt_var_cur, ddt_ddt_var = f(vars)
            # ...
            for node in nodes:
                for sub_order in range(1, order):
                    vname = n_state(ddt(node.name, order=sub_order))
                    cur_vname = ddt(node.name, order=sub_order)
                    stmts.append(
                        ast.Assign(
                            targets=[set_ctx(mk_var(vname), ast.Store)],
                            value=set_ctx(mk_var(cur_vname), ast.Load),
                        )
                    )
                vname = ddt(node.name, order=order)
                if order != 0:
                    vname = n_state(vname)
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
        if not return_jax:
            stmts.append(
                set_ctx(
                    ast.Return(
                        ast.List(
                            [
                                mk_var(n_state(ddt(name, order=1)))
                                for name in input_return_names
                            ]
                        )
                    ),
                    ast.Load,
                )
            )
        else:
            stmts.append(
                set_ctx(
                    ast.Return(
                        ast.Call(
                            ast.Attribute(
                                value=mk_var("jnp"),
                                attr="array",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.List(
                                    [
                                        mk_var(n_state(ddt(name, order=1)))
                                        for name in input_return_names
                                    ]
                                )
                            ],
                            keywords=[],
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
        return ast.FunctionDef(ode_fn_name, arguments, stmts, decorator_list=[]), stmts

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
            mk_arg(self.ATTR_VAL),
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

    def _compile_top_stmt_odeterm(self, stmts: list) -> ast.FunctionDef:
        """Top level statement for ode term

        Args:
            stmts (list): list of statements

        Returns:
            ast.FunctionDef: top level statement
        """

        args = [
            mk_arg(self.TIME_RANGE),
            mk_arg(self.INIT_STATE),
            mk_arg(self.SWITCH_VAL),
            mk_arg(self.ATTR_VAL),
            mk_arg(self.SIM_SEED),
            mk_arg(self.FN_ARGS),
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
