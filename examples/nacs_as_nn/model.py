import equinox as eqx
import jax.numpy as jnp
import numpy as np
import spec
from diffrax import AbstractSolver, Tsit5
from utils import create_connected_grid

from ark.cdg.cdg import CDG, CDGEdge, CDGNode
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import TrainableMgr


def one_clipping(x):
    "clip x to between [-1, 1]"
    return jnp.clip(x, -1, 1)


def locking_fn(x, lock_strength: float):
    "Injection locking function with phase normalize from pi to 1"
    return lock_strength * jnp.sin(2 * jnp.pi * x)


def coupling_fn(x, cpl_strength: float):
    "Oscillator coupling function with phase normalize from pi to 1"
    return cpl_strength * jnp.sin(jnp.pi * x)


class NACSysGrid(eqx.Module):

    dynamical_sys: BaseAnalogCkt
    sys_name: str
    dimension: tuple[int, int]
    neighbor_dist: int
    trainable_mgr: TrainableMgr

    def __init__(
        self,
        sys_name: str,
        n_rows: int,
        n_cols: int,
        neighbor_dist: int,
        solver: AbstractSolver = Tsit5(),
        trainable_initialization: str = "uniform",
    ):

        self.sys_name = sys_name
        self.dimension = (n_rows, n_cols)
        self.neighbor_dist = neighbor_dist

        cdg_spec, cdg, nodes, edges = self.build_system_cdg()
        self.trainable_mgr = self.assign_trainables(
            nodes, edges, trainable_initialization
        )
        nodes_flatten = sum(nodes, [])
        sys_cls = OptCompiler().compile(
            prog_name=sys_name,
            cdg=cdg,
            cdg_spec=cdg_spec,
            trainable_mgr=self.trainable_mgr,
            readout_nodes=nodes_flatten,
            vectorize=True,
            normalize_weight=False,
        )
        self.dynamical_sys = sys_cls(
            init_trainable=self.trainable_mgr.get_initial_vals(),
            is_stochastic=False,
            solver=solver,
        )

    def build_system_cdg(self):
        sys_name = self.sys_name
        if sys_name == "CNN":
            node_type = spec.Neuron
            node_attrs = {"z": 0.0, "act": one_clipping}
            edge_type = spec.Coupling
            edge_attrs = {"k": 0.0}
            bidirectional_edge = True
            cdg_spec = spec.cnn_spec
        elif sys_name == "OBC":
            node_type = spec.Osc
            node_attrs = {
                "lock_fn": locking_fn,
                "osc_fn": coupling_fn,
                "locking_strength": 0.0,
                "cpl_strength": 0.0,
            }
            edge_type = spec.Coupling
            edge_attrs = {"k": 0.0}
            bidirectional_edge = True
            cdg_spec = spec.obc_spec
        elif sys_name == "NNL":
            node_type = spec.Neuron
            node_attrs = {"z": 0.0, "act": jnp.tanh}
            edge_type = spec.Coupling
            edge_attrs = {"k": 0.0}
            bidirectional_edge = False
            cdg_spec = spec.cnn_spec
        else:
            raise ValueError(f"Unknown system name {sys_name}")

        n_rows, n_cols = self.dimension
        cdg, nodes, edges = create_connected_grid(
            n_rows=n_rows,
            n_cols=n_cols,
            node_type=node_type,
            node_attrs=node_attrs,
            edge_type=edge_type,
            edge_attrs=edge_attrs,
            length=self.neighbor_dist,
            bidirectional_edge=bidirectional_edge,
        )
        return cdg_spec, cdg, nodes, edges

    def assign_trainables(
        self,
        nodes: list[list[CDGNode]],
        edges: list[CDGEdge],
        trainable_initialization: str,
    ):
        mgr = TrainableMgr()
        if self.sys_name == "CNN" or self.sys_name == "NNL":
            node_trainable = ["z"]
            edge_trainable = ["k"]
        elif self.sys_name == "OBC":
            node_trainable = ["locking_strength", "cpl_strength"]
            edge_trainable = ["k"]
        else:
            raise ValueError(f"Unknown system name {self.sys_name}")

        if trainable_initialization == "uniform":
            init_fn = lambda: np.random.uniform(-1, 1)
        elif trainable_initialization == "normal":
            init_fn = np.random.randn

        for node_row in nodes:
            for node in node_row:
                for attr in node_trainable:
                    node.attrs[attr] = mgr.new_analog(init_val=init_fn())
        for edge in edges:
            for attr in edge_trainable:
                edge.attrs[attr] = mgr.new_analog(init_val=init_fn())

        return mgr
