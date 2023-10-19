from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    def __init__(self) -> None:
        self.use_wandb, self.wandb_run = False, None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    def _init_wandb(self, **kwargs):
        import wandb

        self.wandb_run = wandb.init(config=self._wandb_config(**kwargs))

    @abstractmethod
    def _wandb_config(self, **kwargs) -> dict[str, Any]:
        pass

    def _wandb_logging(self, data: dict[str, Any]):
        """Log data to wandb.

        Args:
            data (dict[str, Any]): Keys are the names of the metrics.
        """
        self.wandb_run.log(data)
