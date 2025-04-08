from typing import Callable, Literal, Optional

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


def straight_through_quantize(x, n_bits: int):
    """Straight through quantization function.

    Args:
        x (Array): input array
        bits (int): # of bits for quantization

    Returns:
        Array: quantized array
    """
    levels = 2 * n_bits
    # Compute step size: equally spaced between -1 and 1 => step = 2 / (levels - 1)
    step = 2 / (levels - 1)
    # Compute the quantization index:
    # Shift x so that -1 maps to 0 and then divide by step.
    index = jnp.round((x + 1) / step)
    # Map index back to the quantized value.
    x_quant = -1 + index * step
    # Use stop_gradient so that during backpropagation the gradient is as if this were an identity.
    return x + jax.lax.stop_gradient(x_quant - x)


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
        mismatch_rstd (float): standard deviation for the parameter random mismatch. Will affect the bias `z`
            and the coupling strength `k` of the nodes and edges respectively.
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
    mismatch_rstd: float

    def __init__(
        self,
        sys_name: Literal["CNN", "OBC", "CANN"],
        mismatch_rstd: float,
        n_rows: int,
        n_cols: int,
        neighbor_dist: int,
        input_type: Literal["fixed", "initial_state"],
        solver: AbstractSolver = Tsit5(),
        trainable_initialization: Literal["uniform", "normal"] = "uniform",
    ):
        self.sys_name = sys_name
        self.mismatch_rstd = mismatch_rstd
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
            # TODO: Figure out why aggregate line causes system w/ mismatched to run faster but
            # system w/o mismatches to run slower
            aggregate_args_lines=True if mismatch_rstd else False,
        )
        self.dynamical_sys = sys_cls(
            init_trainable=trainable_mgr.get_initial_vals(),
            is_stochastic=False,
            solver=solver,
        )
        self.initial_state_mapping = self._get_initial_state_mapping(
            nodes, input_nodes_edges
        )

    def __call__(self, x: Array, time_info: TimeInfo, args_seed: int) -> Array:
        x = self._forward_preprocess(x)
        trace = self.dynamical_sys(
            time_info=time_info,
            initial_state=x,
            switch=[],  # No switch
            args_seed=args_seed,
            noise_seed=0,  # No random noise
        ).reshape(self.dimension)
        return self._forward_postprocess(trace)

    def _build_system_cdg(self):
        sys_name = self.sys_name
        if sys_name == "CNN" or sys_name == "CANN":
            if self.mismatch_rstd == 0:
                node_type = spec.Neuron
                edge_type = spec.Coupling
            else:
                node_type = spec.Neuron_mismatched
                edge_type = spec.Coupling_mismatched
                node_type.attr_def["z"].rstd = self.mismatch_rstd
                edge_type.attr_def["k"].rstd = self.mismatch_rstd
            node_attrs = {
                "z": 0.0,
                "act": one_clipping if sys_name == "CNN" else wrapped_tanh,
            }
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
            output = one_clipping(trace)
        elif self.sys_name == "OBC":
            output = jnp.sin(jnp.pi * trace)
        elif self.sys_name == "CANN":
            output = jnp.tanh(trace)
        else:
            raise ValueError(f"Unknown system name {self.sys_name}")
        return output


class NACSysClassifier(eqx.Module):
    """Classifier model with novel analog computing system.

    Args:
        n_classes (int): # of classes
        img_size (int): size of the input image
        nacs_sys (Optional[NACSysGrid]): analog computing system
        hidden_size (int): # of hidden neurons
        key (jax.random.PRNGKey): random key for initialization
        img_downsample (int, optional): downsample ratio for the image. Defaults to 1.
        use_batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
        adc_quantization_bits (int, optional): # of bits for the image  quantization prior to the digital model.
            Defaults to None (no quantization).

    """

    batch_norm_hidden: Optional[eqx.nn.BatchNorm]
    fc_out: eqx.nn.Linear
    fc_hidden: eqx.nn.Linear
    nacs_sys: NACSysGrid
    img_downsample: int
    adc_quantization_bits: Optional[int]

    def __init__(
        self,
        n_classes: int,
        img_size: int,
        nacs_sys: Optional[NACSysGrid],
        hidden_size: int,
        key: jax.random.PRNGKey,
        img_downsample: int = 1,
        use_batch_norm: bool = False,
        adc_quantization_bits: Optional[int] = None,
    ):
        assert not nacs_sys or nacs_sys.dimension == (img_size, img_size)
        assert (
            img_size % img_downsample == 0
        ), "img_size must be divisible by downsample ratio"
        key1, key2 = jax.random.split(key, 2)
        self.nacs_sys = nacs_sys
        img_dim = img_size // img_downsample
        self.img_downsample = img_downsample
        self.fc_hidden = eqx.nn.Linear(
            img_dim**2,
            hidden_size,
            key=key1,
            use_bias=False,
        )
        self.fc_out = eqx.nn.Linear(hidden_size, n_classes, key=key2, use_bias=False)
        self.batch_norm_hidden = (
            eqx.nn.BatchNorm(input_size=hidden_size, axis_name="batch")
            if use_batch_norm
            else None
        )
        self.adc_quantization_bits = adc_quantization_bits

    def __call__(
        self, x: Array, state: eqx.nn.State, time_info: TimeInfo, mismatch_seed: int
    ) -> tuple[Array, eqx.nn.State]:
        if self.nacs_sys:
            x = self.nacs_sys(x, time_info, mismatch_seed)
        # downsample with average pooling
        x = eqx.nn.AvgPool2d(
            kernel_size=(self.img_downsample, self.img_downsample),
            stride=(self.img_downsample, self.img_downsample),
        )(x[None, :, :]).squeeze()
        if self.adc_quantization_bits:
            x = straight_through_quantize(x, self.adc_quantization_bits)
        x = self.fc_hidden(x.flatten())
        x = jax.nn.relu(x)
        if self.batch_norm_hidden:
            x, state = self.batch_norm_hidden(x, state)
        return self.fc_out(x), state

    def weight(self):
        return {
            "fc_hidden": self.fc_hidden.weight,
            "fc_out": self.fc_out.weight,
            "nacs_sys": (
                self.nacs_sys.dynamical_sys.weights() if self.nacs_sys else None
            ),
        }


def stack_fn(x, fs, **kwargs):
    return jnp.stack([f(x, kwargs) for f in fs])


class MixedNACSysClassifier(eqx.Module):
    """Classifier model with a mixture of novel analog computing system  .

    Args:
        n_classes (int): # of classes
        img_size (int): size of the input image
        nacs_sys_list (list[NACSysGrid]): analog computing system
        hidden_size (int): # of hidden neurons
        key (jax.random.PRNGKey): random key for initialization
        use_batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
        adc_quantization_bits (int, optional): # of bits for the image  quantization prior to the digital model.
            Defaults to None (no quantization).

    """

    batch_norm_hidden: Optional[eqx.nn.BatchNorm]
    fc_out: eqx.nn.Linear
    fc_hidden: eqx.nn.Linear
    nacs_sys_list: list[NACSysGrid]
    kernel_size: int
    adc_quantization_bits: Optional[int]
    image_size: int

    def __init__(
        self,
        n_classes: int,
        img_size: int,
        nacs_sys_list: list[NACSysGrid],
        hidden_size: int,
        key: jax.random.PRNGKey,
        use_batch_norm: bool = False,
        adc_quantization_bits: Optional[int] = None,
    ):
        nacs_sys_dim = nacs_sys_list[0].dimension
        assert nacs_sys_dim[0] == nacs_sys_dim[1], "nacs system must be square"
        for nacs_sys in nacs_sys_list:
            assert (
                nacs_sys.dimension == nacs_sys_dim
            ), "All nacs systems must have the same dimension"
        kernel_size = nacs_sys_dim[0]
        n_kernels = len(nacs_sys_list)
        assert (
            img_size % kernel_size == 0
        ), "img_size must be divisible by the nacs system kernel"
        key1, key2 = jax.random.split(key, 2)
        self.nacs_sys_list = nacs_sys_list
        input_dim = (img_size // kernel_size) ** 2 * n_kernels
        self.kernel_size = kernel_size
        self.fc_hidden = eqx.nn.Linear(
            input_dim,
            hidden_size,
            key=key1,
            use_bias=False,
        )
        self.fc_out = eqx.nn.Linear(hidden_size, n_classes, key=key2, use_bias=False)
        self.batch_norm_hidden = (
            eqx.nn.BatchNorm(input_size=hidden_size, axis_name="batch")
            if use_batch_norm
            else None
        )
        self.adc_quantization_bits = adc_quantization_bits
        self.image_size = img_size

    def __call__(
        self, x: Array, state: eqx.nn.State, time_info: TimeInfo, mismatch_seed: int
    ) -> tuple[Array, eqx.nn.State]:

        k = self.kernel_size
        m = self.image_size
        x_reshaped = x.reshape(m // k, k, m // k, k)
        x_blocks = x_reshaped.transpose(0, 2, 1, 3)
        x_blocks_flat = x_blocks.reshape(m // k, m // k, k, k)

        # Apply the stack function to each block
        x = jnp.stack(
            [
                jax.vmap(
                    jax.vmap(
                        lambda block: f(block, time_info, mismatch_seed), in_axes=(0)
                    ),
                    in_axes=(0),
                )(x_blocks_flat)
                for f in self.nacs_sys_list
            ]
        )
        x = jnp.mean(x, axis=(3, 4))

        if self.adc_quantization_bits:
            x = straight_through_quantize(x, self.adc_quantization_bits)
        x = self.fc_hidden(x.flatten())
        x = jax.nn.relu(x)
        if self.batch_norm_hidden:
            x, state = self.batch_norm_hidden(x, state)
        return self.fc_out(x), state
