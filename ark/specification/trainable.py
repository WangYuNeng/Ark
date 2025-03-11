"""Handle trainable attributes."""

import jax.numpy as jnp
from jaxtyping import Array


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

    def get_initial_vals(self, datatype: str) -> Array | list:
        """Get the initial values of all trainable parameters."""
        if datatype == "analog":
            return jnp.array([trainable.init_val for trainable in self.analog])
        elif datatype == "digital":
            return [trainable.init_val for trainable in self.digital]
        else:
            raise ValueError(f"Unknown datatype: {datatype}")

    def set_initial_vals(self, datatype: str, vals: Array | list):
        """Set the initial values of all trainable parameters.
        If the datatype is digital, vals should be a list.
        If the datatype is analog, vals should be a jnp array.
        """
        if datatype == "analog":
            assert isinstance(
                vals, jnp.ndarray
            ), "Inital values of analog attributes should be a jnp array."
            for trainable, val in zip(self.analog, vals):
                trainable.init_val = val
        elif datatype == "digital":
            assert isinstance(
                vals, list
            ), "Inital values of digital attributes should be a list."
            for trainable, val in zip(self.digital, vals):
                trainable.init_val = val
        else:
            raise ValueError(f"Unknown datatype: {datatype}")
