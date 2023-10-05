from typing import Any, Callable

import numpy as np

from ark.optimization.optimizer import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing Engine"""

    def __init__(
        self,
        temperature: float,
        frozen_temp: float,
        temp_decay: float,
        inner_iteraion: int,
    ) -> None:
        """Initialize the simulated annealing engine.

        Args:
            temperature (float): The initial temperature.
            frozen_temp (float): The temperature to stop the annealing.
            temp_decay (float): The decay rate of the temperature.
            inner_iteraion (int): The number of iterations for each temperature.
        """
        self.temperature = temperature
        self.frozen_temp = frozen_temp
        self.temp_decay = temp_decay
        self.inner_iteration = inner_iteraion
        self._logging = False
        self._log = {}

    def optimize(
        self,
        init_sol: Any,
        neighbor_func: Callable,
        cost_func: Callable,
        logging: bool = True,
    ) -> Any:
        """Optimize the cost function from the initial solution using simulated annealing.

        Args:
            init_sol (Any): The initial solution to start with.
            neighbor_func (Callable): The function to generate a neighbor solution from
            a given solution.
            cost_func (Callable): The function to calculate the cost of a given solution.
            logging (bool, optional): Whether to log the process. Defaults to True.

        Returns:
            Any (same as init_sol): The minimum-cost solution found in the initial phase.
        """
        self._logging, self._log = logging, self._new_log()
        sol = self.initial_phase(init_sol, neighbor_func, cost_func)
        best_sol = self.annealing_phase(sol, neighbor_func, cost_func)
        return best_sol

    def initial_phase(
        self, init_sol: Any, neighbor_func: Callable, cost_func: Callable
    ) -> Any:
        """Initial searching phase for profiling.

        Args:
            init_sol (Any): The initial solution to start with.
            neighbor_func (Callable): The function to generate a neighbor solution from
            a given solution.
            cost_func (Callable): The function to calculate the cost of a given solution.

        Returns:
            Any (same as init_sol): The minimum-cost solution found in the initial phase.
        """
        return init_sol

    def annealing_phase(
        self, init_sol: Any, neighbor_func: Callable, cost_func: Callable
    ) -> Any:
        """Annealing phase to search for the minimum cost solution.

        Args:
            init_sol (Any): The initial solution to start with.
            neighbor_func (Callable): The function to generate a neighbor solution from
            a given solution.
            cost_func (Callable): The function to calculate the cost of a given solution.

        Returns:
            Any (same as init_sol): The minimum-cost solution found.
        """
        best_sol = cur_sol = init_sol
        best_cost = cur_cost = cost_func(cur_sol)
        temp = self.temperature
        while not self._is_frozen(temp):
            for _ in range(self.inner_iteration):
                neighbor_sol = neighbor_func(cur_sol)
                neighbor_cost = cost_func(neighbor_sol)
                acc = self._accept(cur_cost, neighbor_cost, temp)
                if acc:
                    if cur_cost < best_cost:
                        best_sol = cur_sol
                        best_cost = cur_cost
                    cur_sol = neighbor_sol
                    cur_cost = neighbor_cost
                if self._logging:
                    self._log["temp"].append(temp)
                    self._log["acc"].append(acc)
                    self._log["cost"].append(cur_cost)
            temp = self._update_temp(temp)
        return best_sol

    def _accept(self, cur_cost: float, neighbor_cost: float, temp: float) -> bool:
        """Accept the neighbor solution with a probability.

        Args:
            cur_cost (float): The cost of the current solution.
            neighbor_cost (float): The cost of the neighbor solution.
            temp (float): The current temperature.
        """
        if neighbor_cost < cur_cost:
            return True
        else:
            return np.random.rand() < np.exp(-(neighbor_cost - cur_cost) / temp)

    def _is_frozen(self, temp: float) -> bool:
        return temp < self.frozen_temp

    def _update_temp(self, temp: float) -> float:
        return temp * self.temp_decay

    def _new_log(self) -> dict[str, list]:
        return {"temp": [], "acc": [], "cost": []}
