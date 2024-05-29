import ast
from copy import copy
from typing import Callable

import jax
import jax.numpy as jnp

from ark.cdg.cdg import CDG, CDGElement
from ark.compiler import ArkCompiler, mk_arg, set_ctx
from ark.optimization.base_module import BaseAnalogCkt
from ark.specification.attribute_def import AttrDefMismatch, Trainable
from ark.specification.specification import CDGSpec

ark_compiler = ArkCompiler()


def mk_var_generator(name: str):
    return lambda: ast.Name(id=name)


def base_configure_simulation(self):
    self.t0 = 0
    self.t1 = 1
    self.dt0 = 0.01
    self.y0 = jnp.array([0, 1, 0, 2])
    self.saveat = jnp.linspace(self.t0, self.t1, 11)


def mk_assign(target: ast.expr, value: ast.expr):
    """target = value"""
    return ast.Assign(
        targets=[set_ctx(target, ast.Store)],
        value=set_ctx(value, ast.Load),
    )


def mk_call(fn: ast.expr, args: list[ast.expr]):
    """fn(*args)"""
    return ast.Call(
        func=fn,
        args=args,
        keywords=[],
    )


def mk_list(lst: list[ast.Name | ast.Constant]):
    return ast.List(elts=lst)


def mk_arr_access(lst: ast.Name, idx: ast.expr):
    return ast.Subscript(
        value=lst,
        slice=idx,
    )


def mk_list_val_expr(lst: list):
    """make the list value to be expressions"""
    lst_expr = []
    for val in lst:
        if isinstance(val, ast.expr):
            lst_expr.append(val)
        elif isinstance(val, (int, float)) or val is None:
            lst_expr.append(ast.Constant(value=val))
        else:
            raise ValueError(f"Unknown type {type(val)} to be converted in the list")
    return lst_expr


def mk_jnp_call(args: list, call_fn: str):
    """jnp.call_fn(*args)"""

    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="jnp"), attr=call_fn),
        args=mk_list_val_expr(args),
        keywords=[],
    )


def mk_jax_random_call(args: list, call_fn: str):
    """jax.random.call_fn(*args)"""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="jax"), attr="random"), attr=call_fn
        ),
        args=mk_list_val_expr(args),
        keywords=[],
    )


def mk_jnp_arr_access(arr: ast.Name, idx: ast.expr):
    """arr.at[idx]"""
    return ast.Subscript(
        value=ast.Attribute(value=arr, attr="at"),
        slice=idx,
    )


def mk_jnp_assign(arr: ast.Name, idx: ast.expr, val: ast.Name | ast.Constant):
    """
    arr = arr.at[idx].set(val)
    """
    return mk_assign(
        target=arr,
        value=ast.Call(
            func=ast.Attribute(
                value=mk_jnp_arr_access(copy(arr), idx),
                attr="set",
            ),
            args=[val],
            keywords=[],
        ),
    )


def cnt_n_mismatch_attr(cdg: CDG) -> int:
    cnt = 0
    ele: CDGElement
    for ele in cdg.nodes + cdg.edges:
        for val in ele.attr_def.values():
            if isinstance(val, AttrDefMismatch):
                cnt += 1
    return cnt


class OptCompiler:

    SELF = "self"
    SWITCH = "switch"
    MISMATCH_SEED = "mismatch_seed"
    NOISE_SEED = "noise_seed"
    CLIPPED_MIN, CLIPPED_MAX = -1, 1

    def __init__(self) -> None:
        pass

    def compile(
        self,
        prog_name: str,
        cdg: CDG,
        cdg_spec: CDGSpec,
        trainable_len: int,
        do_normalization: bool = True,
        do_clipping: bool = True,
    ) -> type:
        """Compile the cdg to an equinox.Module.

        Args:
            prog_name (str): name of the program
            cdg (CDG): the dynamical graph
            cdg_spec (CDGSpec): the specification of the dynamical graph
            trainable_len (int): length of the trainable array
            do_normalization (bool, optional): whether to normalize the trainable
            parameters or not. If true, the traianble weights are assumed to within
            [-1, 1]. Defaults to True.
            do_clipping (bool, optional): whether to clip the value within the range
            specified in ``cdg_spec``. Defaults to True.

        Returns:
            type: the compiled module.
        """

        (ode_term, noise_term), node_mapping, switch_map, num_attr_map, fn_attr_map = (
            ark_compiler.compile_odeterm(cdg, cdg_spec)
        )

        self.mm_used_idx = 0
        namespace = {"jax": jax, "jnp": jnp}

        # Usefule constants
        args_len = len(switch_map) + sum(len(x) for x in num_attr_map.values())
        fargs_len = sum(len(x) for x in fn_attr_map.values())
        n_mismatch = cnt_n_mismatch_attr(cdg)

        # Input variables ast expr
        self_expr_gen = mk_var_generator(self.SELF)
        switch_expr_gen = mk_var_generator(self.SWITCH)
        mismatch_seed_expr_gen = mk_var_generator(self.MISMATCH_SEED)
        noise_seed_expr_gen = mk_var_generator(self.NOISE_SEED)

        #  Function arguments
        fn_args = [None for _ in range(fargs_len)]

        # Common variables ast expr
        args_expr_gen = mk_var_generator("args")
        trainable_expr_gen = mk_var_generator("trainable")
        mm_prng_key_expr_gen = mk_var_generator("mm_key")
        mm_arr_expr_gen = mk_var_generator("mm_arr")

        stmts = []
        # Assign self.trainable to trainable for readability
        # and clipping the trainable values
        # trainable = jnp.clip(self.trainable, CLIPPED_MIN, CLIPPED_MAX)
        clipped_trainable = mk_jnp_call(
            args=[
                ast.Attribute(value=self_expr_gen(), attr="trainable"),
                self.CLIPPED_MIN,
                self.CLIPPED_MAX,
            ],
            call_fn="clip",
        )
        stmts.append(
            mk_assign(
                trainable_expr_gen(),
                clipped_trainable,
            )
        )
        # Initialize the jnp arrays
        init_arr = mk_jnp_call(args=[args_len], call_fn="zeros")
        stmts.append(mk_assign(args_expr_gen(), init_arr))

        # Initialize the mismatch array
        stmts.append(
            mk_assign(
                mm_prng_key_expr_gen(),
                mk_jax_random_call(args=[mismatch_seed_expr_gen()], call_fn="PRNGKey"),
            )
        )
        mismatch_arr = mk_jax_random_call(
            args=[mm_prng_key_expr_gen(), mk_list(mk_list_val_expr([n_mismatch]))],
            call_fn="normal",
        )
        stmts.append(
            mk_assign(
                mm_arr_expr_gen(),
                mismatch_arr,
            )
        )
        ele: CDGElement
        for ele in cdg.nodes + cdg.edges:
            assert ele.name in num_attr_map or ele.name in fn_attr_map
            if ele.name in fn_attr_map:
                fn_attr_to_idx = fn_attr_map[ele.name]
            if ele.name in num_attr_map:
                num_attr_to_idx = num_attr_map[ele.name]
            for attr, val in ele.attrs.items():
                if isinstance(val, Callable):
                    # Handle function attribute
                    assert attr in fn_attr_to_idx
                    fn_args[fn_attr_to_idx[attr]] = val
                else:
                    assert attr in num_attr_to_idx
                    if isinstance(val, Trainable):
                        # Handle trainable attribute
                        val: Trainable
                        trainable_id = val.idx
                        val_expr = mk_arr_access(
                            trainable_expr_gen(), ast.Constant(value=trainable_id)
                        )
                    else:
                        # Handle fixed attribute
                        val_expr = ast.Constant(value=val)

                    # If the attribute is mismatched, apply the mismatch
                    if isinstance(ele.attr_def[attr], AttrDefMismatch):
                        val_expr = self._mk_mismatch_expr(
                            orig_expr=val_expr,
                            mm_arr_expr=mm_arr_expr_gen(),
                            attr_def=ele.attr_def[attr],
                        )

                    stmts.append(
                        mk_jnp_assign(
                            arr=args_expr_gen(),
                            idx=ast.Constant(value=num_attr_to_idx[attr]),
                            val=val_expr,
                        )
                    )

        # Return the args
        stmts.append(set_ctx(ast.Return(value=args_expr_gen()), ast.Load))
        # Compile the statements to make_args(self, switch, mismatch_seed) function
        make_args_fn = ast.FunctionDef(
            name="make_args",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    mk_arg(self.SELF),
                    mk_arg(self.SWITCH),
                    mk_arg(self.MISMATCH_SEED),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=stmts,
            decorator_list=[],
        )
        module = ast.Module([make_args_fn], type_ignores=[])
        module = ast.fix_missing_locations(module)
        print(ast.unparse(module))
        exec(
            compile(source=module, filename="tmp.py", mode="exec"),
            namespace,
        )

        ode_fn = lambda self, t, y, args: ode_term(t, y, args, fn_args)
        noise_fn = lambda self, t, y, args: noise_term(t, y, args, fn_args)
        configure_simulation = base_configure_simulation

        opt_module = type(
            prog_name,
            (BaseAnalogCkt,),
            {
                "ode_fn": ode_fn,
                "noise_fn": noise_fn,
                "make_args": namespace["make_args"],
                "configure_simulation": configure_simulation,
            },
        )

        return opt_module

    def _mk_mismatch_expr(
        self, orig_expr: ast.expr, mm_arr_expr: ast.Name, attr_def: AttrDefMismatch
    ):
        """
        Return the expression after adding the mismatch
        """
        if attr_def.std:
            # orig_expr + std*mm_arr_expr[self.mm_used_idx])
            std_expr = ast.Constant(value=attr_def.std)
            mm_expr = ast.BinOp(
                left=orig_expr,
                op=ast.Add(),
                right=ast.BinOp(
                    left=std_expr,
                    op=ast.Mult(),
                    right=mk_arr_access(
                        mm_arr_expr, ast.Constant(value=self.mm_used_idx)
                    ),
                ),
            )
        elif attr_def.rstd:
            # orig_expr * (1 + std*mm_arr_expr[self.mm_used_idx])
            rstd_expr = ast.Constant(value=attr_def.rstd)
            mm_expr = ast.BinOp(
                left=orig_expr,
                op=ast.Mult(),
                right=ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.Add(),
                    right=ast.BinOp(
                        left=rstd_expr,
                        op=ast.Mult(),
                        right=mk_arr_access(
                            mm_arr_expr, ast.Constant(value=self.mm_used_idx)
                        ),
                    ),
                ),
            )
        else:
            raise ValueError("Must specify either rstd or std")

        self.mm_used_idx += 1
        return mm_expr
