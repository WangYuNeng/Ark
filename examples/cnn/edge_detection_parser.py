from argparse import ArgumentParser

parser = ArgumentParser()

# Example command: python pattern_recog_digit.py --gauss_std 0.1 --trans_noise_std 0.1
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default="one-to-one")
parser.add_argument(
    "--weight_init",
    type=str,
    default="edge-detection",
    choices=["edge-detection", "random"],
    help="Method to initialize training weights.",
)
parser.add_argument(
    "--mismatched_node",
    action="store_true",
    help="Use 10 percent random mismatched node for the CNN",
)
parser.add_argument(
    "--mismatched_edge",
    type=float,
    default=0.0,
    help="Use random mismatched edge with specified rstd for the CNN",
)
parser.add_argument(
    "--activation",
    type=str,
    default="ideal",
    choices=["ideal", "diffpair"],
    help="Activation function to use in the CNN",
)

parser.add_argument(
    "--end_time", type=float, default=1.0, help="End time of simulation"
)
parser.add_argument(
    "--n_time_points",
    type=int,
    default=100,
    help="Number of time points in the simulation",
)
parser.add_argument(
    "--store_edge_detection",
    action="store_true",
    help="Iterate over the training and testing data to generate edge detected images with current cnn templates",
)
parser.add_argument(
    "--ed_img_path",
    type=str,
    help="Path to the edge detected images. If store_edge_detection is True, this path will be used to save the edge detected images",
)
parser.add_argument(
    "--downsample",
    type=int,
    default=1,
    help="Downsample the images by this factor",
)
parser.add_argument("--steps", type=int, default=32, help="Number of training steps")
parser.add_argument("--bz", type=int, default=512, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
parser.add_argument(
    "--optimizer", type=str, default="adam", help="Type of the optimizer"
)
parser.add_argument(
    "--plot_evolve",
    type=int,
    default=0,
    help="Number of time points to plot the evolution",
)
parser.add_argument("--num_plot", type=int, default=4, help="Number of samples to plot")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument(
    "--dataset",
    type=str,
    default="simple",
    choices=["simple", "mnist"],
    help="Dataset to use",
)
args = parser.parse_args()
