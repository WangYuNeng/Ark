from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import spec
from diffrax import AbstractSolver, Tsit5
from jaxtyping import Array
from utils import create_connected_grid

from ark.cdg.cdg import CDGEdge, CDGNode
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


def wrapped_tanh(x):
    return jnp.tanh(x)


class NACSysGrid(eqx.Module):
    """Novel Analog Computing System (NACS) grid system.

    The system implements one of the analog computing paradigm, CNN, OBC, or CANN and serves
    as a "layer" in the neural network. The system is constructed in a grid topology with
    neighboring connection (in a square with specified `neighbor_dist`).

    The system's input can either be "fixed" or "initial_state". In the "fixed" input, the
    input is fed with an additional node and fixed over time. In the "initial_state" input,
    the input is fed with setting the initial states of the grid state variables and the state
    will evolve over time.

    Args:
        sys_name (str): name of the system, one of "CNN", "OBC", "CANN"
        n_rows (int): # of rows in the grid
        n_cols (int): # of rows in the grid
        neighbor_dist (int): distance of the neighboring connection, If the sid length is an even number,
            the value specifies the side length of the sqaure; otherwise, the value specifies the half diagonal
            length of the square.
        input_type (str): type of input, one of "fixed", "initial_state"
        solver (AbstractSolver, optional): solver for numerical integration. Defaults to Tsit5().
        trainable_initialization (str, optional): intialization method for the trainable parmeters.
            Defaults to "uniform".
    """

    dynamical_sys: BaseAnalogCkt
    sys_name: str
    input_type: str
    dimension: tuple[int, int]
    initial_state_mapping: list[int]
    neighbor_dist: int

    def __init__(
        self,
        sys_name: Literal["CNN", "OBC", "CANN"],
        n_rows: int,
        n_cols: int,
        neighbor_dist: int,
        input_type: Literal["fixed", "initial_state"],
        solver: AbstractSolver = Tsit5(),
        trainable_initialization: Literal["uniform", "normal"] = "uniform",
    ):
        self.sys_name = sys_name
        self.input_type = input_type
        self.dimension = (n_rows, n_cols)
        self.neighbor_dist = neighbor_dist

        cdg_spec, cdg, nodes, edges, input_nodes_edges = self._build_system_cdg()
        trainable_mgr = self._assign_trainables(
            nodes, edges, input_nodes_edges, trainable_initialization
        )
        nodes_flatten = sum(nodes, [])
        sys_cls = OptCompiler().compile(
            prog_name=sys_name,
            cdg=cdg,
            cdg_spec=cdg_spec,
            trainable_mgr=trainable_mgr,
            readout_nodes=nodes_flatten,
            vectorize=True,
            normalize_weight=False,
            do_clipping=False,
        )
        self.dynamical_sys = sys_cls(
            init_trainable=trainable_mgr.get_initial_vals(),
            is_stochastic=False,
            solver=solver,
        )
        self.initial_state_mapping = self._get_initial_state_mapping(
            nodes, input_nodes_edges
        )

    def __call__(self, x: Array, time_info: TimeInfo) -> Array:
        x = self._forward_preprocess(x)
        trace = self.dynamical_sys(
            time_info=time_info,
            initial_state=x,
            switch=[],  # No switch
            args_seed=0,  # No random mismatch
            noise_seed=0,  # No random noise
        ).T
        return self._forward_postprocess(trace)

    def _build_system_cdg(self):
        sys_name = self.sys_name
        if sys_name == "CNN" or sys_name == "CANN":
            node_type = spec.Neuron
            node_attrs = {
                "z": 0.0,
                "act": one_clipping if sys_name == "CNN" else wrapped_tanh,
            }
            edge_type = spec.Coupling
            edge_attrs = {"k": 0.0}
            bidirectional_edge = False
            cdg_spec = spec.cnn_spec if sys_name == "CNN" else spec.cann_spec
        elif sys_name == "OBC":
            node_type = spec.Osc
            node_attrs = {
                "lock_fn": locking_fn,
                "osc_fn": coupling_fn,
                "lock_strength": 0.0,
                "cpl_strength": 0.0,
            }
            edge_type = spec.Coupling
            edge_attrs = {"k": 0.0}
            bidirectional_edge = True
            cdg_spec = spec.obc_spec
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

        if self.input_type == "fixed":
            input_nodes_edges = [[None for _ in range(n_cols)] for _ in range(n_rows)]
            for row, node_row in enumerate(nodes):
                for col, node in enumerate(node_row):
                    # Add input node
                    inp_node = spec.Inp()
                    inp_edge = edge_type(**edge_attrs)
                    cdg.connect(inp_edge, inp_node, node)
                    input_nodes_edges[row][col] = (inp_node, inp_edge)
        elif self.input_type == "initial_state":
            input_nodes_edges = None
        else:
            raise ValueError(f"Unknown input type {self.input_type}")
        return cdg_spec, cdg, nodes, edges, input_nodes_edges

    def _assign_trainables(
        self,
        nodes: list[list[CDGNode]],
        edges: list[CDGEdge],
        input_nodes_edges: list[list[tuple[CDGNode, CDGEdge]]],
        trainable_initialization: str,
    ):
        mgr = TrainableMgr()
        if self.sys_name == "CNN" or self.sys_name == "CANN":
            node_trainable = ["z"]
            edge_trainable = ["k"]
        elif self.sys_name == "OBC":
            node_trainable = ["lock_strength", "cpl_strength"]
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

        if input_nodes_edges:
            for row in input_nodes_edges:
                for _, inp_edge in row:
                    for attr in edge_trainable:
                        inp_edge.attrs[attr] = mgr.new_analog(init_val=init_fn())

        return mgr

    def _get_initial_state_mapping(
        self,
        grid_nodes: list[list[CDGNode]],
        input_nodes_edges: list[list[tuple[CDGNode, CDGEdge]]],
    ) -> list[int]:
        """Get the mapping to transform the input vector x to the initial state vector

        Args:
            grid_nodes (list[list[CDGNode]]): the grid nodes
            input_nodes_edges (list[list[tuple[CDGNode, CDGEdge]]]): the input nodes and edges
        Return:
            arr (list[int]): i'th element of flattened input vector x should be mapped to arr[i]
            position of the initial state vector.
        """
        inp_node: CDGNode
        init_state_mapping = []
        if input_nodes_edges:
            for row in input_nodes_edges:
                for inp_node, _ in row:
                    init_state_mapping.append(
                        self.dynamical_sys.node_to_init_state_id(inp_node.name)
                    )

        else:
            for row in grid_nodes:
                for node in row:
                    init_state_mapping.append(
                        self.dynamical_sys.node_to_init_state_id(node.name)
                    )
        return init_state_mapping

    def _forward_preprocess(self, x: Array) -> Array:
        """Reorder the input vector x to the initial state vector.

        If the input type is "fixed", the initial state vector is all zeros except the input nodes mapped
        from the input vector x. If the input type is "initial_state", the initial state vector is the mapped
        input vector x.

        Args:
            x (Array): input vector

        Returns:
            Array: initial state vector
        """
        x = x.flatten()
        jnp_state_map = jnp.array(self.initial_state_mapping)
        if self.input_type == "fixed":
            init_state_dim = 2 * self.dimension[0] * self.dimension[1]
        elif self.input_type == "initial_state":
            init_state_dim = self.dimension[0] * self.dimension[1]
        else:
            raise ValueError(f"Unknown input type {self.input_type}")
        init_state = jnp.zeros(init_state_dim)
        init_state = init_state.at[jnp_state_map].set(x)
        return init_state

    def _forward_postprocess(self, trace: Array) -> Array:
        """Postprocess the trace to get proper output

        CNN: The output is clipped to [-1, 1]
        OBC: The output is mapped cyclically to [-1, 1]
        CANN: The output pass a tanh function
        """

        if self.sys_name == "CNN":
            return one_clipping(trace)
        elif self.sys_name == "OBC":
            return jnp.sin(jnp.pi * trace)
        elif self.sys_name == "CANN":
            return jnp.tanh(trace)
        else:
            raise ValueError(f"Unknown system name {self.sys_name}")


class NACSysClassifier(eqx.Module):
    """Classifier model with novel analog comuting system.

    Args:
        n_classes (int): # of classes
        img_size (int): size of the input image
        nacs_sys (Optional[NACSysGrid]): analog computing system
    """

    w_out: Array
    nacs_sys: NACSysGrid

    def __init__(
        self,
        n_classes: int,
        img_size: int,
        nacs_sys: Optional[NACSysGrid],
    ):
        assert not nacs_sys or nacs_sys.dimension == (img_size, img_size)
        self.nacs_sys = nacs_sys
        self.w_out = jnp.array(np.random.randn(n_classes, img_size**2))

    def __call__(self, img: Array, time_info: TimeInfo) -> Array:
        if self.nacs_sys:
            x = self.nacs_sys(img, time_info)
        else:
            x = img.flatten()
        return jnp.matmul(self.w_out, x)

    def weight(self):
        return {
            "w_out": self.w_out.copy(),
            "nacs_sys": (
                self.nacs_sys.dynamical_sys.weights() if self.nacs_sys else None
            ),
        }
