import argparse

parser = argparse.ArgumentParser()

# System configuration
parser.add_argument(
    "--input_type", type=str, choices=["initial_state", "fixed"], default="fixed"
)
parser.add_argument(
    "--input_kernel_size",
    type=int,
    default=1,
    help="Kernel size for input-to-reservoir coupling",
)
parser.add_argument(
    "--grid_kernel_size",
    type=int,
    default=3,
    help="Kernel size for reservoir internal coupling",
)

# Simulation parameters
parser.add_argument(
    "--n_readout",
    type=int,
    default=1,
    help="Number of readout points. Points are equally spaced between 0 and the total simulation time"
    "(at least include the final time point).",
)
parser.add_argument(
    "--t_end", type=float, default=10.0, help="Total simulation cycles."
)

# Training parameters
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--image_downsample",
    type=int,
    default=1,
    help="Downsample factor for images. Applied before the linear layer.",
)
parser.add_argument("--n_epochs", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--validation_split", type=float, default=0.1)
parser.add_argument("--early_stopping", type=int, default=100)
parser.add_argument("--testing", action="store_true")
parser.add_argument("--test_only", action="store_true")

# Data parameters
parser.add_argument(
    "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"]
)

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--load_path", type=str, default="")
parser.add_argument("--save_path", type=str, default="")


args = parser.parse_args()
