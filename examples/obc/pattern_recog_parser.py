from argparse import ArgumentParser

parser = ArgumentParser()

# Example command: python pattern_recog_main.py --gauss_std 0.1 --trans_noise_std 0.1
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default="one-to-one")
parser.add_argument(
    "--matrix_solve",
    action="store_true",
    help="Use matrix form of the OBC to solve the ODEs instead of the Ark compiled version",
)
parser.add_argument(
    "--pattern_shape",
    type=str,
    default="5x3",
    choices=["5x3", "10x6"],
    help="Shape of the digit patterns, currently support 5x3 and 10x6",
)
parser.add_argument(
    "--connection",
    type=str,
    default="neighbor",
    choices=["neighbor", "all"],
    help="Connection pattern between the oscillators",
)
parser.add_argument(
    "--trainable_connection",
    action="store_true",
    help="Whether the connections (ON/OFF) between oscillators are trainable",
)
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
    "--fix_coupling_weight",
    action="store_true",
    help="Fix the coupling weight to the initial value",
)
parser.add_argument(
    "--trainable_locking",
    action="store_true",
    help="Whether the locking strength is trainable",
)
parser.add_argument(
    "--locking_strength",
    type=float,
    default=1.0,
    help="Initial strength of the locking",
)
parser.add_argument(
    "--trainable_coupling",
    action="store_true",
    help="Whether the total coupling strength is trainable",
)
parser.add_argument(
    "--coupling_strength",
    type=float,
    default=1.0,
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
    "--l1_norm_weight", type=float, default=0.0, help="L1 norm weight for the loss"
)
parser.add_argument(
    "--point_per_cycle", type=int, default=50, help="Number of time points per cycle"
)
parser.add_argument(
    "--snp_prob",
    type=float,
    default=0.0,
    help="Salt-and-pepper noise of the initial image",
)
parser.add_argument(
    "--gauss_std",
    type=float,
    default=0.0,
    help="Gaussian noise std of the initial image",
)
parser.add_argument(
    "--uniform_noise",
    action="store_true",
    help="Add uniform noise to the initial image",
)
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
parser.add_argument("--tag", type=str, default=None, help="Tag for the wandb run")
parser.add_argument(
    "--save_weight", type=str, default=None, help="Path to save weights"
)
parser.add_argument(
    "--load_weight", type=str, default=None, help="Path to load weights"
)
parser.add_argument("--test", action="store_true", help="Test the model")
parser.add_argument("--test_bz", type=int, default=1024, help="Size of the test batch")
parser.add_argument(
    "--weight_drop_ratio",
    type=float,
    default=0.0,
    help="Ratio of smallest weights (abs) to drop during testing",
)
parser.add_argument(
    "--test_seed", type=int, default=428, help="Random seed for testing"
)
parser.add_argument(
    "--vectorize_odeterm",
    action="store_true",
    help="Whether to compile the ODE term in vectorized form",
)
args = parser.parse_args()
