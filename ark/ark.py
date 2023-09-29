"""Wrapper class for the ARK framework"""
from types import FunctionType
from typing import Optional

from ark.cdg.cdg import CDG, AttrImpl, CDGEdge, CDGElement, CDGExecutionData, CDGNode
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.specification.specification import CDGSpec
from ark.validator import VALID, ArkValidator


class Ark:
    """Wrapper class for the ARK framework

    Attributes:
        validator: ArkValidator
        compiler: ArkCompiler
        cdg_spec: CDGSpec
    """

    def __init__(
        self,
        validator: Optional[ArkValidator] = ArkValidator(solver=SMTSolver()),
        compiler: Optional[ArkCompiler] = ArkCompiler(rewrite=RewriteGen()),
        cdg_spec: Optional[CDGSpec] = None,
    ) -> None:
        """Initialize the ARK framework

        Args:
            validator (ArkValidator, optional): Validator to use.
            Defaults to ArkValidator(solver=SMTSolver()).
            compiler (ArkCompiler, optional): Compiler to use.
            Defaults to ArkCompiler(rewrite=RewriteGen()).
            cdg_spec (CDGSpec, optional): The specification of the CDG.
            Defaults to None.
        """
        self._validator = validator
        self._compiler = compiler
        self._cdg_spec = cdg_spec
        self._prog, self._node_mapping, self._edge_mapping, self._attr_mapping = (
            None,
            None,
            None,
            None,
        )

    def validate(self, cdg: CDG) -> bool:
        """Validate the given CDG against the specification

        Args:
            cdg (CDG): The CDG to validate

        Returns:
            bool: True if the CDG is valid, False otherwise
        """
        flag, _ = self.validator.validate(cdg, self.cdg_spec)
        if flag != VALID:
            return False
        return True

    def compile(
        self,
        cdg: CDG,
        verbose: int = 0,
    ) -> None:
        """Compile the given CDG to a program

        Args:
            cdg (CDG): The CDG to compile
        """
        (
            self._prog,
            self._node_mapping,
            self._edge_mapping,
            self._attr_mapping,
        ) = self.compiler.compile(cdg, self.cdg_spec, verbose)

    def execute(
        self,
        cdg: Optional[CDG] = None,
        cdg_execution_data: Optional[CDGExecutionData] = None,
        time_eval: list[float] = None,
        init_seed: Optional[int] = None,
        sim_seed: int = 0,
        store_inplace: bool = True,
        **kwargs,
    ) -> None | dict[CDGNode, list[list[float]]]:
        """Execute the compiled program with the given CDG or CDG data

        The traces of the stateful nodes are stored CDGNodes if store_inplace is True.
        Otherwise, the traces will be returned as a dictionary.
        Execution from CDG data and not stored in-place is to support multiprocessing.

        Args:
            cdg (CDG, optional): The CDG to execute.
            cdg_execution_data (CDGExecutionData, optional): The CDG data to execute.
            time_eval (list[float]): time points to evaluate.
            init_seed (int, optional): Seed for intialize mismatched attributes.
            Defaults to 0.
            sim_seed (int, optional): Seed for ode simulation. Defaults to 0.
            store_inplace (bool, optional): Store the execution results in cdg.
            Defaults to True.
        Returns:
            None | dict[CDGNode, list[list[float]]]: The mapping from stateful nodes
            to execution traces if store_inplace is False.
        """
        assert self._prog is not None, "Program is not compiled."
        if not cdg and not cdg_execution_data:
            raise ValueError("Either CDG or CDG data must be provided.")
        if cdg:
            if cdg_execution_data:
                Warning("CDG provided. CDG data is ignored.")
            cdg_execution_data = cdg.execution_data(seed=init_seed)
        node_to_init_val, switch_to_val, element_to_attr = cdg_execution_data
        init_states = self._map_init_state(node_to_init_val)
        switch_vals = self._map_switch_val(switch_to_val)
        attr_vals = self._map_attr_val(element_to_attr)
        sol = self._prog(
            time_range=[time_eval[0], time_eval[-1]],
            init_states=init_states,
            switch_vals=switch_vals,
            attr_vals=attr_vals,
            sim_seed=sim_seed,
            t_eval=time_eval,
            **kwargs,
        )
        if store_inplace:
            if not cdg:
                raise ValueError("CDG must be provided to store the results in-place.")
            self._store_exec_results(cdg, sol)
        else:
            return {
                node: [
                    sol.y[self._node_mapping[node] + order]
                    for order in range(node.order)
                ]
                for node in self._node_mapping
            }

    def print_prog(self) -> None:
        """Print the compiled program"""
        self._compiler.print_prog()

    def dump_prog(self, file_name: str) -> None:
        """Dump the compiled program to the given file

        Args:
            file_name (str): The file name to dump the compiled program
        """
        self._compiler.dump_prog(file_name)

    @property
    def validator(self) -> ArkValidator:
        return self._validator

    @validator.setter
    def validator(self, validator: ArkValidator) -> None:
        self._validator = validator

    @property
    def compiler(self) -> ArkCompiler:
        return self._compiler

    @compiler.setter
    def compiler(self, compiler: ArkCompiler) -> None:
        self._compiler = compiler

    @property
    def cdg_spec(self) -> CDGSpec:
        return self._cdg_spec

    @cdg_spec.setter
    def cdg_spec(self, cdg_spec: CDGSpec) -> None:
        self._cdg_spec = cdg_spec

    def _store_exec_results(self, cdg: CDG, sol) -> None:
        """Store the execution results to the CDG

        Args:
            cdg (CDG): The CDG to store the execution results
            sol (list[list[float]]): The execution results
        """
        assert self._node_mapping is not None, "Node mapping is not constructed."
        for node in cdg.stateful_nodes:
            for order in range(node.order):
                node.set_trace(n=order, trace=sol.y[self._node_mapping[node] + order])

    def _map_init_state(
        self, node_to_init_state: dict[CDGNode, list]
    ) -> list[int | float]:
        """Map the initial state values to the corresponding position

        Args:
            node_to_init_state (dict[CDGNode, list]): The mapping from nodes to the mapping
            from order to initial state values.

        Returns:
            init_states (list[int | float]): list of initial state values
            arranged for the compiled ode simulation
        """
        n_states = sum(
            [len(init_states) for init_states in node_to_init_state.values()]
        )
        assert self._node_mapping is not None, "Node mapping is not constructed."
        assert node_to_init_state.keys() == self._node_mapping.keys()
        init_states = [0 for _ in range(n_states)]
        for node in node_to_init_state:
            init_vals = node_to_init_state[node]
            for order, val in enumerate(init_vals):
                init_states[self._node_mapping[node] + order] = val
        return init_states

    def _map_switch_val(self, switch_to_val: dict[CDGEdge, bool]) -> list[int | float]:
        """Map the switch values to the corresponding position

        Args:
            switch_to_val (dict[CDGEdge, bool]): The mapping from switches to values.

        Returns:
            list[int | float]: list of switch values
            arranged for the compiled ode simulation
        """
        n_switch = len(switch_to_val)
        assert self._edge_mapping is not None, "Edge mapping is not constructed."
        assert n_switch == len(self._edge_mapping)
        switch_vals = [0 for _ in range(n_switch)]
        for switch, val in switch_to_val.items():
            switch_vals[self._edge_mapping[switch]] = val
        return switch_vals

    def _map_attr_val(
        self, element_to_attr_sample: dict[CDGElement, dict[str, AttrImpl]]
    ) -> list[int | float | FunctionType]:
        """Map the attribute values to the corresponding position

        Args:
            element_to_attr_sample (dict[CDGElement, dict[str, AttrImpl]]): The mapping
            from elements to the mapping from attributes to values.

        Returns:
            list[int | float | FunctionType]: list of attribute values
            arranged for the compiled ode simulation
        """

        assert self._attr_mapping is not None, "Attribute mapping is not constructed."
        assert len(element_to_attr_sample) == len(self._attr_mapping)
        attr_vals = [
            0 for _ in range(sum(len(attrs) for attrs in self._attr_mapping.values()))
        ]
        for ele, attrs in element_to_attr_sample.items():
            for attr, val in attrs.items():
                attr_vals[self._attr_mapping[ele][attr]] = val
        return attr_vals
