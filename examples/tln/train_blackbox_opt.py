# Check if ax and cma are installed
try:
    from ax import Client, RangeParameterConfig
except ImportError:
    Client = None
    RangeParameterConfig = None
    cma = None

from functools import partial
from typing import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import wandb
from ark.optimization.base_module import BaseAnalogCkt


def evaluate_model_wrapper(
    data: jax.Array,
    *args,
    model: BaseAnalogCkt,
    loss_fn: Callable,
):
    a_trainable = jnp.array(args)
    model = eqx.tree_at(lambda m: m.a_trainable, model, a_trainable)
    loss = loss_fn(model, *data)
    return loss.item()


def train_ax(
    model: BaseAnalogCkt,
    precise_loss: Callable,
    dataloader: Generator,
    steps: int,
    use_wandb: bool,
):
    if not Client:
        raise ImportError(
            "ax is not installed, cannot run hyperparameter optimization."
        )
    client = Client(random_seed=np.random.randint(0, 2**31))
    # Define six float parameters x1, x2, x3, ... for the Hartmann6 function, which is typically evaluated on the unit hypercube
    param_keys = (
        ["lc0", "gr0"]
        + [f"lc{i}" for i in range(1, 5)]
        + [f"gr{i}" for i in range(1, 5)]
        + [f"lc{i}" for i in range(5, 9)]
        + [f"gr{i}" for i in range(5, 9)]
        + [f"gm{i}" for i in range(16)]
    )  # LC and GR parameters
    parameters = (
        [
            RangeParameterConfig(name=f"lc{i}", parameter_type="float", bounds=(-1, 1))
            for i in range(9)
        ]
        + [
            RangeParameterConfig(name=f"gr{i}", parameter_type="float", bounds=(-1, 1))
            for i in range(9)
        ]
        + [
            RangeParameterConfig(name=f"gm{i}", parameter_type="float", bounds=(-1, 1))
            for i in range(16)
        ]
    )

    client.configure_experiment(parameters=parameters)
    metric_name = "loss"
    objective = f"-{metric_name}"
    client.configure_optimization(objective=objective)

    # Initial point for optimization
    a_trainable = model.a_trainable
    initial_points = {"lc0": a_trainable[0].item(), "gr0": a_trainable[1].item()}
    initial_points.update({f"lc{i}": a_trainable[i + 1].item() for i in range(1, 5)})
    initial_points.update({f"gr{i}": a_trainable[i + 5].item() for i in range(1, 5)})
    initial_points.update({f"lc{i}": a_trainable[i + 5].item() for i in range(5, 9)})
    initial_points.update({f"gr{i}": a_trainable[i + 9].item() for i in range(5, 9)})
    initial_points.update({f"gm{i}": a_trainable[i + 18].item() for i in range(16)})

    eval_model = partial(
        evaluate_model_wrapper,
        model=model,
        loss_fn=precise_loss,
    )
    eqx.filter_jit(eval_model)

    # Add initial point to the experiment
    trial_index = -1

    for step, data in zip(range(steps), dataloader):
        if trial_index == -1:
            trial_index = client.attach_trial(parameters=initial_points)
            parameters = initial_points
        else:
            trial_index, parameters = list(
                client.get_next_trials(max_trials=1).items()
            )[0]

        param_flatten = [parameters[key] for key in param_keys]
        print(param_flatten)
        loss = eval_model(
            data,
            *param_flatten,
        )
        raw_data = {
            metric_name: loss,
        }
        client.complete_trial(
            trial_index=trial_index,
            raw_data=raw_data,
        )
        print(f"Completed trial {trial_index} with {raw_data=}")
        if use_wandb:
            wandb.log(
                {
                    "train_loss": loss,
                    "loss_precise": loss,
                }
            )

    best_parameters, prediction, index, name = client.get_best_parameterization(
        use_model_predictions=False
    )
    loss_best = prediction[metric_name][0]
    weight_best = (
        jnp.array([best_parameters[key] for key in param_keys]),
        [],
    )

    return loss_best, weight_best
