from argparse import ArgumentParser

parser = ArgumentParser()

# Example command: python pattern_recog_digit.py --gauss_std 0.1 --trans_noise_std 0.1
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default="one-to-one")
parser.add_argument(
    "--n_cycle",
    type=int,
    default=1,
    help="Number of cycles to wait for the oscillators to read out",
)
parser.add_argument(
    "--weight_init",
    type=str,
    default="hebbian",
    choices=["hebbian", "random"],
    help="Method to initialize training weights.",
)
parser.add_argument(
    "--weight_bits",
    type=int,
    default=None,
    help="Number of bits for the digital coupling strength, None for analog coupling",
)
parser.add_argument(
    "--gumbel_temp_start",
    type=float,
    default=50,
    help="Initial temperature for the gumbel softmax",
)
parser.add_argument(
    "--gumbel_temp_end",
    type=float,
    default=1,
    help="Final temperature for the gumbel softmax",
)
parser.add_argument(
    "--hard_gumbel",
    action="store_true",
    help="Use hard gumbel softmax and straight-through estimator. Default is soft gumbel softmax",
)
parser.add_argument(
    "--gumbel_schedule",
    type=str,
    default="linear",
    choices=["linear", "exp"],
    help="Annealing schedule for the gumbel temperature",
)
parser.add_argument(
    "--trainable_locking",
    action="store_true",
    help="Whether the locking strength is trainable",
)
parser.add_argument(
    "--locking_strength", type=float, default=1, help="Initial strength of the locking"
)
parser.add_argument(
    "--trainable_coupling",
    action="store_true",
    help="Whether the total coupling strength is trainable",
)
parser.add_argument(
    "--coupling_strength",
    type=float,
    default=1,
    help="Initial strength of the coupling",
)
parser.add_argument(
    "--diff_fn",
    type=str,
    choices=["periodic_mse", "periodic_mean_max_se", "normalize_angular_diff"],
    default="periodic_mean_max_se",
    help="The function to evaluate the difference between the readout and the target",
)
parser.add_argument(
    "--point_per_cycle", type=int, default=50, help="Number of time points per cycle"
)
parser.add_argument("--snp_prob", type=float, default=0.0, help="Salt-and-pepper noise")
parser.add_argument("--gauss_std", type=float, default=0.0, help="Gaussian noise std")
parser.add_argument(
    "--trans_noise_std", type=float, default=0.0, help="Transition noise std"
)
parser.add_argument(
    "--n_class", type=int, default=5, help="Number of classes to recognize"
)
parser.add_argument("--steps", type=int, default=32, help="Number of training steps")
parser.add_argument("--bz", type=int, default=512, help="Batch size")
parser.add_argument(
    "--validation_split", type=float, default=0.5, help="Validation split ratio"
)
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
parser.add_argument(
    "--no_noiseless_train", action="store_true", help="Skip noiseless training"
)
parser.add_argument("--wandb", action="store_true", help="Log to wandb")

args = parser.parse_args()
