import argparse

parser = argparse.ArgumentParser()

# System configuration
parser.add_argument("--sys_name", type=str, choices=["OBC", "CNN", "CANN"])
parser.add_argument("--input_type", type=str, choices=["initial_state", "fixed"])
parser.add_argument("--neighbor_dist", type=int, default=2)
parser.add_argument("--trainable_init", type=str, choices=["uniform", "normal"])

# Simulation parameters
parser.add_argument("--readout_time", type=float, default=1.0)
parser.add_argument("--n_time_points", type=int, default=20)

# Training parameters
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--validation_split", type=float, default=0.1)
parser.add_argument("--testing", action="store_true")

# Data parameters
parser.add_argument(
    "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"]
)

parser.add_argument("--wandb", action="store_true")


args = parser.parse_args()
