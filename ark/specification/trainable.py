"""Handle trainable attributes."""

import jax.numpy as jnp


class Trainable:
    """Trainable attribute.

    Attributes:
        idx: The index in the args array.
        (A temporary solution to allow shared weight)
    """

    def __init__(self, idx: int, init_val=None) -> None:
        self.idx = idx
        self._init_val = init_val

    def __str__(self) -> str:
        return "Trainable"

    @property
    def init_val(self):
        if isinstance(self._init_val, (int, float)):
            return self._init_val

        elif isinstance(self._init_val, list):
            return jnp.array(self._init_val)

        elif isinstance(self._init_val, jnp.ndarray):
            return self._init_val

        else:
            raise ValueError(f"Unknown type of init_val: {type(self._init_val)}")

    @init_val.setter
    def init_val(self, val):
        self._init_val = val


class TrainableMgr:
    """Manage indexing and mapping the trainable parameters."""

    def __init__(self) -> None:
        self._analog: list[Trainable] = []
        self._digital: list[Trainable] = []

    def reset(self) -> None:
        """Reset the trainable parameters."""
        self._analog = []
        self._digital = []

    def new_analog(self, init_val=None) -> Trainable:
        """Add a trainable analog parameter."""
        return self._new_trainable(self.analog, init_val)

    def new_digital(self, init_val=None) -> Trainable:
        """Add a trainable digital parameter."""
        return self._new_trainable(self.digital, init_val)

    @property
    def digital(self) -> list[Trainable]:
        """Get the list of digital trainable parameters."""
        return self._digital

    @property
    def analog(self) -> list[Trainable]:
        """Get the list of analog trainable parameters."""
        return self._analog

    def _new_trainable(self, param_list: str, init_val=None) -> Trainable:
        """Add a trainable parameter."""
        trainable = Trainable(len(param_list), init_val=init_val)
        param_list.append(trainable)
        return trainable

    def get_initial_vals(self, datatype: str):
        """Get the initial values of all trainable parameters."""
        if datatype == "analog":
            return jnp.array([trainable.init_val for trainable in self.analog])
        elif datatype == "digital":
            return [trainable.init_val for trainable in self.digital]
        else:
            raise ValueError(f"Unknown datatype: {datatype}")
