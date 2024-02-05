import argparse
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differentiable_sspuf import SwitchableStarPUF
from tqdm import tqdm
from train import I2O_chls, i2o_loss, step

parser = argparse.ArgumentParser()
parser.add_argument("--n_branch", type=int, default=10)
parser.add_argument("--line_len", type=int, default=4)
parser.add_argument("--n_order", type=int, default=40)
parser.add_argument("--readout_time", type=float, default=10e-9)
parser.add_argument("--rand_init", action="store_true")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--chl_per_inst", type=int, default=64)
parser.add_argument("--inst_per_batch", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--logistic_k", type=float, default=40)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()


N_BRANCH = args.n_branch
LINE_LEN = args.line_len
N_ORDER = args.n_order
READOUT_TIME = args.readout_time
RAND_INIT = args.rand_init
BATCH_SIZE = args.batch_size
CHL_PER_INST = args.chl_per_inst
INST_PER_BATCH = args.inst_per_batch
STEPS = args.steps
LOGISTIC_K = args.logistic_k
OUTPUT = args.output


# Example Testing values
init_vals_20ns_start = {
    "gm_c": jnp.array(
        [
            [0.85353027, 0.87160655],
            [1.36489997, 1.15324941],
            [0.50969158, 1.24936502],
            [0.9693642, 0.57403478],
        ]
    ),
    "gm_l": jnp.array(
        [
            [1.21544818, 0.70646345],
            [0.87212753, 1.1356638],
            [0.8040117, 0.86603637],
            [0.80724017, 0.90369365],
        ]
    ),
    "c_val": jnp.array([1.28969255, 1.05447954, 0.79314443, 1.01151726, 0.79523754]),
    "l_val": jnp.array([1.33134152, 1.14411907, 1.32640049, 1.00130298]),
}

init_vals_20ns_end = {
    "gm_c": jnp.array(
        [
            [1.1570814, 1.1750473],
            [1.3934822, 1.1844265],
            [0.51233846, 1.2382622],
            [1.0086144, 0.62439924],
        ]
    ),
    "gm_l": jnp.array(
        [
            [1.0904831, 0.5954707],
            [1.3466474, 1.6121832],
            [0.7885809, 0.8490631],
            [0.5105633, 0.6046546],
        ]
    ),
    "c_val": jnp.array([0.82953715, 1.0941348, 1.0346369, 1.0068684, 0.9403395]),
    "l_val": jnp.array([1.370831, 0.94048023, 1.3692745, 1.2418798]),
}


model = SwitchableStarPUF(
    n_branch=N_BRANCH,
    line_len=LINE_LEN,
    n_order=N_ORDER,
    init_vals=init_vals_20ns_end,
)

print(model.gm_c, model.gm_l)
print(model.c_val, model.l_val)

loader = I2O_chls(
    INST_PER_BATCH, CHL_PER_INST, N_BRANCH, model.lds_a_shape, READOUT_TIME
)
i2o_ideal = partial(i2o_loss, quantize_fn=step)
jax.jit(i2o_ideal)

loss_vals = []
for i, (switches, mismatch, t) in tqdm(enumerate(loader), total=STEPS):
    if i == STEPS:
        break
    loss_vals.append(i2o_ideal(model, switches, mismatch, t))

pickle.dump(loss_vals, open(f"{OUTPUT}.pkl", "wb"))

loss_vals = jnp.array(loss_vals)
plt.hist(loss_vals, bins=20)
plt.title(
    f"{OUTPUT} I2O Score Distribution, Mean: {jnp.mean(loss_vals):.2f}, Median: {jnp.median(loss_vals):.2f}"
)
plt.xlabel("I2O Score")
plt.ylabel("Count")
plt.savefig(f"{OUTPUT}.png", dpi=300)
plt.show()
