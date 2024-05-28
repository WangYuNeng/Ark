import ast
from typing import Callable

import jax.numpy as jnp

from ark.cdg.cdg import CDG, CDGElement
from ark.compiler import ArkCompiler, mk_var, set_ctx
from ark.optimization.base_module import BaseAnalogCkt
from ark.specification.attribute_def import AttrDefMismatch, Trainable
from ark.specification.specification import CDGSpec

ark_compiler = ArkCompiler()


def base_configure_simulation(self):
    self.t0 = 0
    self.t1 = 1
    self.dt0 = 0.01
    self.y0 = jnp.array([0, 1, 0, 2])
    self.saveat = jnp.linspace(self.t0, self.t1, 11)


def mk_assign(target: ast.Name, value: ast.Expr):
    """target = value"""
    return ast.Assign(
        targets=[set_ctx(target, ast.Store)],
        value=set_ctx(value, ast.Load),
    )


def mk_call(fn: ast.Expr, args: list[ast.Expr]):
    """fn(*args)"""
    return ast.Call(
        func=fn,
        args=args,
        keywords=[],
    )


def set_list_val_expr(lst: list):
    """set the list value to be expressions"""
    lst_expr = []
    for val in lst:
        if isinstance(val, (ast.Name, ast.Constant)):
            lst_expr.append(val)
        elif isinstance(val, (int, float)):
            lst_expr.append(ast.Constant(value=val))
        else:
            raise ValueError(f"Unknown type {type(val)} to be converted in the list")
    return lst_expr


def mk_jnp_call(args: list, call_fn: str):
    """jnp.call_fn(*args)"""

    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="jnp"), attr=call_fn),
        args=set_list_val_expr(args),
        keywords=[],
    )


def mk_jax_random_call(args: list, call_fn: str):
    """jax.random.call_fn(*args)"""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="jax"), attr="random"), attr=call_fn
        ),
        args=set_list_val_expr(args),
        keywords=[],
    )


def mk_jnp_arr_access(arr: ast.Name, idx: ast.Expr):
    """arr.at[idx]"""
    return ast.Subscript(
        value=ast.Attribute(value=arr, attr="at"),
        slice=idx,
    )


def mk_jnp_assign(arr: ast.Name, idx: ast.Expr, val: ast.Name | ast.Constant):
    """
    arr = arr.at[idx].set(val)
    """
    return mk_assign(
        target=arr,
        value=ast.Call(
            func=ast.Attribute(
                value=mk_jnp_arr_access(arr, idx),
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

    def __init__(self) -> None:
        pass

    def compile(
        self,
        prog_name: str,
        cdg: CDG,
        cdg_spec: CDGSpec,
        trainable_len: int,
    ) -> type:

        ode_term, node_mapping, switch_map, num_attr_map, fn_attr_map = (
            ark_compiler.compile_odeterm(cdg, cdg_spec)
        )

        # Usefule constants
        args_len = len(switch_map) + sum(len(x) for x in num_attr_map.values())
        n_mismatch = cnt_n_mismatch_attr(cdg)

        # Input variables ast expr
        self_expr = mk_var("self")
        switch_expr = mk_var("switch")
        mismatch_seed_expr = mk_var("mismatch_seed")
        noise_seed_expr = mk_var("noise_seed")

        # Common variables ast expr
        args_expr = mk_var("args")
        fn_args_expr = mk_var("fn_args")
        trainable_expr = mk_var("trainable")
        mm_prng_key_expr = mk_var("mm_key")
        mm_arr_expr = mk_var("mm_arr")

        stmts = []
        # assign self.trainable to trainable for readability
        stmts.append(
            mk_assign(trainable_expr, ast.Attribute(value=self_expr, attr="trainable"))
        )
        # Initialize the jnp arrays
        init_arr = mk_jnp_call(args=[args_len], call_fn="zeros")
        stmts.append(mk_assign(args_expr, init_arr))

        # Initialize the mismatch array
        stmts.append(
            mk_assign(
                mm_prng_key_expr,
                mk_jax_random_call(args=[mismatch_seed_expr], call_fn="PRNGKey"),
            )
        )
        mismatch_arr = mk_jax_random_call(
            args=[mm_prng_key_expr, n_mismatch], call_fn="normal"
        )
        stmts.append(
            mk_assign(
                mm_arr_expr,
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
                    assert attr in fn_attr_to_idx
                    pass
                assert attr in num_attr_to_idx
                if isinstance(val, Trainable):
                    pass
                else:
                    stmts.append(
                        mk_jnp_assign(
                            arr=args_expr,
                            idx=ast.Constant(value=num_attr_to_idx[attr]),
                            val=ast.Constant(value=val),
                        )
                    )

        for stmt in stmts:
            print(ast.unparse(ast.fix_missing_locations(stmt)))

        ode_fn = lambda self, t, y, args: ode_term(t, y, args, None)
        configure_simulation = base_configure_simulation
        rescale_params = lambda self, x: x
        map_params = lambda self, x: x
        clip_params = lambda self, x: x
        add_mismatch = lambda self, x, y: x
        combine_args = lambda self, x, y: x + y

        opt_module = type(
            prog_name,
            (BaseAnalogCkt,),
            {
                "ode_fn": ode_fn,
                "configure_simulation": configure_simulation,
                "rescale_params": rescale_params,
                "map_params": map_params,
                "clip_params": clip_params,
                "add_mismatch": add_mismatch,
                "combine_args": combine_args,
            },
        )

        return opt_module
