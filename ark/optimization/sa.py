from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ark.optimization.optimizer import BaseOptimizer
from ark.util import inspect_func_name


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
        super().__init__()
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
        use_wandb: bool = False,
        meta_data: dict[str, Any] = None,
        check_point_func: Callable = None,
        greedy: bool = False,
    ) -> Any:
        """Optimize the cost function from the initial solution using simulated annealing.

        Args:
            init_sol (Any): The initial solution to start with.
            neighbor_func (Callable): The function to generate a neighbor solution from
                a given solution.
            cost_func (Callable): The function to calculate the cost of a given solution.
            logging (bool, optional): Whether to log the process. Defaults to True.
            use_wandb (bool, optional): Whether to use weight & bias to log the process.
                Defaults to False.
            meta_data (dict[str, Any], optional): The meta data to log with wandb.
                Defaults to None.
            check_point_func (Callable, optional): The function to call when storing a
            checkpoint. The function takes the solution and the cost as the arguments.
                Defaults to None.
            greedy (bool, optional): Whether to use greedy search, meaning only
                accepting a solution when the cost is strictly smaller.
                Defaults to False.

        Returns:
            Any (same as init_sol): The minimum-cost solution found in the initial phase.
        """
        self.use_wandb = use_wandb
        self.check_point_func = check_point_func
        self.greedy = greedy
        if use_wandb:
            self._init_wandb(
                neighbor_func=inspect_func_name(neighbor_func),
                cost_func=inspect_func_name(cost_func),
                check_point_func=inspect_func_name(check_point_func),
                **meta_data,
            )
        self._logging, self._log = logging, self._new_log()
        sol = self._initial_phase(init_sol, neighbor_func, cost_func)
        best_sol = self._annealing_phase(sol, neighbor_func, cost_func)
        if use_wandb:
            self.wandb_run.finish()
        return best_sol

    def visualize_log(self) -> None:
        """Plot the log of the annealing process."""
        if not self._logging:
            raise RuntimeError("No log to visualize")
        # plot all the logs in a 2x3 grid
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].plot(self._log["temp"])
        axs[0, 0].set_title("Temperature")
        axs[0, 1].plot(self._log["acc"], ".")
        axs[0, 1].set_title("Acceptance")
        axs[0, 2].plot(self._log["acc_prob"])
        axs[0, 2].set_title("Acceptance Probability")
        axs[1, 0].plot(self._log["cost"])
        axs[1, 0].set_title("Cost")
        axs[1, 1].plot(self._log["best_cost"])
        axs[1, 1].set_title("Best Cost")
        axs[1, 2].plot(self._log["delta_cost"])
        axs[1, 2].set_title("Delta Cost")
        plt.tight_layout()
        plt.show()

    def _initial_phase(
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

    def _annealing_phase(
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
        if self.check_point_func is not None:
            self.check_point_func(best_sol, best_cost)
        temp = self.temperature
        for _ in tqdm(range(self._frozen_iteration())):
            for _ in range(self.inner_iteration):
                neighbor_sol = neighbor_func(cur_sol)
                neighbor_cost = cost_func(neighbor_sol)
                acc = self._accept(cur_cost, neighbor_cost, temp)
                if self._logging:
                    self._log["temp"].append(temp)
                    self._log["acc"].append(acc)
                    self._log["acc_prob"].append(
                        self._accept_prob(cur_cost, neighbor_cost, temp)
                    )
                    self._log["cost"].append(cur_cost)
                    self._log["best_cost"].append(best_cost)
                    self._log["delta_cost"].append(neighbor_cost - cur_cost)

                if acc:
                    if cur_cost < best_cost:
                        best_sol, best_cost = cur_sol, cur_cost
                        if self.check_point_func is not None:
                            self.check_point_func(best_sol, best_cost)

                    cur_sol, cur_cost = neighbor_sol, neighbor_cost
            if self.use_wandb:
                data = {
                    f"avg_{key}": np.mean(val[-self.inner_iteration :])
                    for key, val in self._log.items()
                } | {
                    f"std_{key}": np.std(val[-self.inner_iteration :])
                    for key, val in self._log.items()
                }
                self._wandb_logging(data)
            temp = self._update_temp(temp)
        return best_sol

    def _accept(self, cur_cost: float, neighbor_cost: float, temp: float) -> bool:
        """Accept the neighbor solution with a probability.

        Args:
            cur_cost (float): The cost of the current solution.
            neighbor_cost (float): The cost of the neighbor solution.
            temp (float): The current temperature.
        """
        return np.random.rand() < self._accept_prob(
            cur_cost=cur_cost, neighbor_cost=neighbor_cost, temp=temp
        )

    def _accept_prob(self, cur_cost: float, neighbor_cost: float, temp: float) -> float:
        """Calculate the acceptance probability of the neighbor solution.

        Args:
            cur_cost (float): The cost of the current solution.
            neighbor_cost (float): The cost of the neighbor solution.
            temp (float): The current temperature.
        """
        if neighbor_cost < cur_cost:
            return 1
        else:
            if self.greedy:
                return 0
            return np.exp(-(neighbor_cost - cur_cost) / temp)

    def _wandb_config(self, **kwargs) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "temperature_decay": self.temp_decay,
            "frozen_temperature": self.frozen_temp,
            "inner_iteration": self.inner_iteration,
            "greedy": self.greedy,
        } | kwargs

    def _frozen_iteration(self) -> int:
        return int(
            np.log(self.frozen_temp / self.temperature) / np.log(self.temp_decay)
        )

    def _is_frozen(self, temp: float) -> bool:
        return temp < self.frozen_temp

    def _update_temp(self, temp: float) -> float:
        return temp * self.temp_decay

    def _new_log(self) -> dict[str, list]:
        return {
            "temp": [],
            "acc": [],
            "cost": [],
            "acc_prob": [],
            "best_cost": [],
            "delta_cost": [],
        }
