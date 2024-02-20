import argparse
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differentiable_sspuf import SSPUF_ODE, SSPUF_MatrixExp
from tqdm import tqdm
from train import I2O_chls, i2o_loss, step

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["MatrixExp", "ODE"])
parser.add_argument("--n_branch", type=int, default=10)
parser.add_argument("--line_len", type=int, default=4)
parser.add_argument("--n_order", type=int, default=40)
parser.add_argument("--n_time_points", type=int, default=100)
parser.add_argument("--readout_time", type=float, default=10e-9)
parser.add_argument("--rand_init", action="store_true")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--chl_per_bit", type=int, default=64)
parser.add_argument("--inst_per_batch", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--logistic_k", type=float, default=40)
parser.add_argument("--weight_file", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

MODEL = args.model
N_BRANCH = args.n_branch
LINE_LEN = args.line_len
N_ORDER = args.n_order
N_TIME_POINTS = args.n_time_points
READOUT_TIME = args.readout_time
RAND_INIT = args.rand_init
BATCH_SIZE = args.batch_size
CHL_PER_BIT = args.chl_per_bit
INST_PER_BATCH = args.inst_per_batch
STEPS = args.steps
LOGISTIC_K = args.logistic_k
WEIGHT_FILE = args.weight_file
OUTPUT = args.output


# Example Testing values
init_vals = pickle.load(open(WEIGHT_FILE, "rb"))

if MODEL == "MatrixExp":
    model = SSPUF_MatrixExp(
        n_branch=N_BRANCH,
        line_len=LINE_LEN,
        n_order=N_ORDER,
        random=RAND_INIT,
        init_vals=init_vals,
    )
elif MODEL == "ODE":
    model = SSPUF_ODE(
        n_branch=N_BRANCH,
        line_len=LINE_LEN,
        n_time_point=N_TIME_POINTS,
        random=RAND_INIT,
        init_vals=init_vals,
    )
else:
    raise ValueError("Unknown model")

print(model.gm_c, model.gm_l)
print(model.c_val, model.l_val)

loader = I2O_chls(
    INST_PER_BATCH, CHL_PER_BIT, N_BRANCH, model.mismatch_len, READOUT_TIME
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
