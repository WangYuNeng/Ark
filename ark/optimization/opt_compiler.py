import jax.numpy as jnp

from ark.cdg.cdg import CDG, CDGElement
from ark.compiler import ArkCompiler
from ark.optimization.base_module import BaseAnalogCkt
from ark.specification.attribute_def import Trainable
from ark.specification.specification import CDGSpec

ark_compiler = ArkCompiler()


def base_configure_simulation(self):
    self.t0 = 0
    self.t1 = 1
    self.dt0 = 0.01
    self.y0 = jnp.array([0, 1, 0, 2])
    self.saveat = jnp.linspace(self.t0, self.t1, 11)


class OptCompiler:

    def __init__(self) -> None:
        pass

    def compile(
        self,
        prog_name: str,
        cdg: CDG,
        cdg_spec: CDGSpec,
    ) -> type:

        ode_term, node_mapping, switch_map, num_attr_map, fn_attr_map = (
            ark_compiler.compile_odeterm(cdg, cdg_spec)
        )

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

    def _initialize_args(
        self,
        cdg: CDG,
        switch_map: dict[str, int],
        num_attr_map: dict[str, dict[str, int]],
    ):
        """Initialize the args for the ode term.

        Initiliaze non-training arguments with values in the cdg.
        Indentify the trainable argument indices.
        """

        args_len = len(switch_map) + sum(len(x) for x in num_attr_map.values())
        args = [None for _ in range(args_len)]
        trainable_idx = []
        mismatch_idx = []

        for sw, val in cdg.switch_to_val.items():
            assert sw in switch_map
            if isinstance(val, Trainable):
                trainable_idx.append(switch_map[sw])
        ele: CDGElement
        for ele in cdg.nodes + cdg:
            assert ele.name in num_attr_map
            attr_to_idx = num_attr_map[ele.name]
            for attr, val in ele.attrs.items():
                assert attr in attr_to_idx
                if isinstance(val, Trainable):
                    trainable_idx.append(num_attr_map[attr])
                else:
                    args[attr_to_idx[attr]] = val
