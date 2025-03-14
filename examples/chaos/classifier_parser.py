import argparse

parser = argparse.ArgumentParser()

# Lorenz96 parameters
parser.add_argument("--n_state_var", type=int, default=4)
parser.add_argument("--forcing", type=float, default=2.0)

# Simulation parameters
parser.add_argument("--readout_time", type=float, default=1.0)
parser.add_argument("--n_time_points", type=int, default=20)
parser.add_argument("--vectorize", action="store_true")

# Training parameters
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--validation_split", type=float, default=0.1)
parser.add_argument("--testing", action="store_true")

# Model parameters
parser.add_argument("--batch_norm", action="store_true")
parser.add_argument("--no_lorenz", action="store_true")

# Data parameters
parser.add_argument(
    "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"]
)

parser.add_argument("--wandb", action="store_true")


args = parser.parse_args()
