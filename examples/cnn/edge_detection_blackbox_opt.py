# Check if ax and cma are installed
try:
    import cma
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
    a_corner,
    a_edge,
    a_center,
    b_corner,
    b_edge,
    b_center,
    bias,
    data: jax.Array,
    model: BaseAnalogCkt,
    activation: Callable,
    loss_fn: Callable,
):
    a_trainable = jnp.array(
        [a_corner, a_edge, a_center, b_corner, b_edge, b_center, bias]
    )
    model = eqx.tree_at(lambda m: m.a_trainable, model, a_trainable)
    loss = loss_fn(model, *data, activation)
    return loss.item()


def train_ax(
    model: BaseAnalogCkt,
    activation: Callable,
    loss_fn: Callable,
    train_dl: Generator,
    steps: int,
    use_wandb: bool,
    limited_range: bool = False,
):
    if not Client:
        raise ImportError(
            "ax is not installed, cannot run hyperparameter optimization."
        )
    client = Client(random_seed=np.random.randint(0, 2**31))
    # Define six float parameters x1, x2, x3, ... for the Hartmann6 function, which is typically evaluated on the unit hypercube
    if limited_range:
        parameters = [
            RangeParameterConfig(
                name="a_corner", parameter_type="float", bounds=(-2, 2)
            ),
            RangeParameterConfig(name="a_edge", parameter_type="float", bounds=(-2, 2)),
            RangeParameterConfig(
                name="a_center", parameter_type="float", bounds=(-2, 2)
            ),
            RangeParameterConfig(
                name="b_corner", parameter_type="float", bounds=(-2, 2)
            ),
            RangeParameterConfig(name="b_edge", parameter_type="float", bounds=(-2, 2)),
            RangeParameterConfig(
                name="b_center", parameter_type="float", bounds=(6, 10)
            ),
            RangeParameterConfig(name="bias", parameter_type="float", bounds=(-2, 2)),
        ]
    else:
        parameters = [
            RangeParameterConfig(
                name="a_corner", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(
                name="a_edge", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(
                name="a_center", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(
                name="b_corner", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(
                name="b_edge", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(
                name="b_center", parameter_type="float", bounds=(-10, 10)
            ),
            RangeParameterConfig(name="bias", parameter_type="float", bounds=(-10, 10)),
        ]

    client.configure_experiment(parameters=parameters)
    metric_name = "loss"
    objective = f"-{metric_name}"
    client.configure_optimization(objective=objective)

    # Initial point for optimization
    a_trainable = model.a_trainable
    initial_points = {
        "a_corner": a_trainable[0].item(),
        "a_edge": a_trainable[1].item(),
        "a_center": a_trainable[2].item(),
        "b_corner": a_trainable[3].item(),
        "b_edge": a_trainable[4].item(),
        "b_center": a_trainable[5].item(),
        "bias": a_trainable[6].item(),
    }

    eval_model = partial(
        evaluate_model_wrapper,
        model=model,
        activation=activation,
        loss_fn=loss_fn,
    )
    eqx.filter_jit(eval_model)

    # Add initial point to the experiment
    trial_index = -1

    for step in range(steps):
        for data in train_dl:
            if trial_index == -1:
                trial_index = client.attach_trial(parameters=initial_points)
                parameters = initial_points
            else:
                trial_index, parameters = list(
                    client.get_next_trials(max_trials=1).items()
                )[0]

            a_corner = parameters["a_corner"]
            a_edge = parameters["a_edge"]
            a_center = parameters["a_center"]
            b_corner = parameters["b_corner"]
            b_edge = parameters["b_edge"]
            b_center = parameters["b_center"]
            bias = parameters["bias"]
            loss = eval_model(
                a_corner=a_corner,
                a_edge=a_edge,
                a_center=a_center,
                b_corner=b_corner,
                b_edge=b_edge,
                b_center=b_center,
                bias=bias,
                data=data,
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
                    data={
                        "train_loss": loss,
                    },
                )

    best_parameters, prediction, index, name = client.get_best_parameterization(
        use_model_predictions=False
    )
    a_corner = best_parameters["a_corner"]
    a_edge = best_parameters["a_edge"]
    a_center = best_parameters["a_center"]
    b_corner = best_parameters["b_corner"]
    b_edge = best_parameters["b_edge"]
    b_center = best_parameters["b_center"]
    bias = best_parameters["bias"]

    loss_best = prediction[metric_name][0]
    weight_best = (
        jnp.array([a_corner, a_edge, a_center, b_corner, b_edge, b_center, bias]),
        [],
    )

    return loss_best, weight_best
