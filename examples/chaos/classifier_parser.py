import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_state_var", type=int, default=10)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument("--readout_time", type=float, default=1.0)
parser.add_argument("--n_time_points", type=int, default=20)

parser.add_argument("--forcing", type=float, default=8.0)

args = parser.parse_args()
