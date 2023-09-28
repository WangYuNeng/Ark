"""Wrapper class for the ARK framework"""
from typing import Optional
from ark.validator import ArkValidator, VALID
from ark.solver import SMTSolver
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG


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
        self._prog, self._node_mapping, self._edge_mapping = None, None, None

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
        import_lib: Optional[dict] = None,
        inline: bool = False,
        verbose: int = 0,
    ) -> None:
        """Compile the given CDG to a program

        Args:
            cdg (CDG): The CDG to compile
        """
        if import_lib is None:
            import_lib = {}
        self._prog, self._node_mapping, self._edge_mapping = self.compiler.compile(
            cdg, self.cdg_spec, import_lib, verbose, inline
        )

    def execute(
        self,
        cdg: CDG,
        time_eval: list[float],
        init_seed: Optional[int] = 0,
        sim_seed: Optional[int] = 0,
        **kwargs,
    ) -> None:
        """Execute the compiled program with the given CDG

        The traces of the stateful nodes will be stored CDGNodes

        Args:
            cdg (CDG): The CDG to execute
            time_eval (list[float]): time points to evaluate
            init_seed (int, optional): Seed for intialize mismatched attributes.
            Defaults to 0.
            sim_seed (int, optional): Seed for ode simulation. Defaults to 0.
        """
        assert self._prog is not None, "Program is not compiled."
        init_states = self._map_init_state(cdg)
        switch_vals = self._map_switch_val(cdg)
        sol = self._prog(
            time_range=[time_eval[0], time_eval[-1]],
            init_states=init_states,
            switch_vals=switch_vals,
            init_seed=init_seed,
            sim_seed=sim_seed,
            t_eval=time_eval,
            **kwargs,
        )
        self._store_exec_results(cdg, sol)

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
        for node in cdg.stateful_nodes():
            for order in range(node.order):
                node.set_trace(n=order, trace=sol.y[self._node_mapping[node] + order])

    def _map_init_state(self, cdg: CDG) -> list[int | float]:
        """Map the initial state values to the corresponding position

        Args:
            cdg (CDG): The input CDG contains the initial state values

        Returns:
            init_states (list[int | float]): list of initial state values
            arranged for the compiled ode simulation
        """
        n_states = cdg.total_1st_order_states()
        statefule_nodes = cdg.stateful_nodes()
        assert self._node_mapping is not None, "Node mapping is not constructed."
        assert len(statefule_nodes) == len(self._node_mapping)
        init_states = [0 for _ in range(n_states)]
        for node in statefule_nodes:
            node_order = node.order
            for order in range(node_order):
                init_states[self._node_mapping[node] + order] = node.init_val(order)
        return init_states

    def _map_switch_val(self, cdg: CDG) -> list[int | float]:
        """Map the switch values to the corresponding position

        Args:
            cdg (CDG): The input CDG contains the switch values

        Returns:
            list[int | float]: list of switch values
            arranged for the compiled ode simulation
        """
        switches = cdg.switches
        assert self._edge_mapping is not None, "Edge mapping is not constructed."
        assert len(switches) == len(self._edge_mapping)
        switch_vals = [0 for _ in range(len(switches))]
        for switch in switches:
            switch_vals[self._edge_mapping[switch]] = switch.val
        return switch_vals
