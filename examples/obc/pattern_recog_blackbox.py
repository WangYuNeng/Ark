# Check if ax and cma are installed
try:
    from ax import ChoiceParameterConfig, Client, RangeParameterConfig
except ImportError:
    Client = None
    RangeParameterConfig = None
    ChoiceParameterConfig = None

from functools import partial
from typing import Callable, Generator, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from spec_optimization import nbits_to_val_choices

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
    # The latter 3 are dummy arguments just for compatibility
    loss = loss_fn(model, *data, gumbel_temp=1.0, hard_gumbel=True, l1_norm_weight=0)
    return loss.item()


def train_ax(
    model: BaseAnalogCkt,
    loss_fn: Callable,
    steps: int,
    batch_size: int,
    dataloader: Generator,
    use_wandb: bool,
    trainable_locking: bool = False,
    trainable_coupling: bool = False,
    weight_bits: Optional[int] = None,
    log_prefix: str = "",
):
    if not Client:
        raise ImportError(
            "ax is not installed, cannot run hyperparameter optimization."
        )
    client = Client(random_seed=np.random.randint(0, 2**31))
    param_keys, parameters = [], []
    n_cpl_param = len(model.a_trainable)
    initial_points = {}
    if trainable_locking:
        param_keys.append("locking")
        parameters.append(
            RangeParameterConfig(
                name="locking", parameter_type="float", bounds=(-10, 10)
            )
        )
        n_cpl_param -= 1
        initial_points["locking"] = model.a_trainable[0].item()

    if trainable_coupling:
        param_keys.append("coupling")
        parameters.append(
            RangeParameterConfig(
                name="coupling", parameter_type="float", bounds=(-10, 10)
            )
        )
        n_cpl_param -= 1
        initial_points["coupling"] = model.a_trainable[1].item()
    param_keys = [f"cpl{i}" for i in range(n_cpl_param)]
    if weight_bits is not None:
        parameters.extend(
            [
                RangeParameterConfig(
                    name=f"cpl{i}", parameter_type="int", bounds=(0, 2**weight_bits - 1)
                )
                for i in range(n_cpl_param)
            ]
        )
    else:
        parameters.extend(
            [
                RangeParameterConfig(
                    name=f"cpl{i}", parameter_type="float", bounds=(-10, 10)
                )
                for i in range(n_cpl_param)
            ]
        )

    client.configure_experiment(parameters=parameters)
    metric_name = "loss"
    objective = f"-{metric_name}"
    client.configure_optimization(objective=objective)

    # Initial point for optimization
    a_trainable = model.a_trainable
    cpl_start_idx = len(a_trainable) - n_cpl_param
    initial_points.update(
        {f"cpl{i}": a_trainable[cpl_start_idx + i].item() for i in range(n_cpl_param)}
    )

    eval_model = partial(
        evaluate_model_wrapper,
        model=model,
        loss_fn=loss_fn,
    )
    eqx.filter_jit(eval_model)

    # Add initial point to the experiment
    trial_index = -1

    for step, data in zip(range(steps), dataloader(batch_size)):
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
                data={
                    f"{log_prefix}_train_loss": loss,
                },
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
