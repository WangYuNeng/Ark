import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=428,
    help="Random seed.",
)
parser.add_argument(
    "--t1",
    type=float,
    default=10.0,
    help="The time duration for the simulation.",
)
parser.add_argument(
    "--dt0",
    type=float,
    default=0.01,
    help="The time step size for the simulation.",
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for the dataloader.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=60,
    help="Number of training steps.",
)
parser.add_argument(
    "--task",
    type=str,
    choices=["3var7clauses", "from_cnf", "random"],
    required=True,
    help="The task to run: '3var7clauses' for a predefined 3-SAT problem with 7"
    " clauses with exact one solution, 'from_cnf' to load SAT problems from a directory"
    " of CNF files., or 'random' to generate random SAT problems.",
)
parser.add_argument(
    "--cnf_dir",
    type=str,
    default=None,
    help="Directory containing CNF files for the 'from_cnf' task.",
)
parser.add_argument(
    "--n_vars",
    type=int,
    default=None,
    help="Number of variables in the SAT problem (only used for 'random' task).",
)
parser.add_argument(
    "--n_clauses",
    type=int,
    default=None,
    help="Number of clauses in the SAT problem (only used for 'random' task).",
)

parser.add_argument(
    "--load_path",
    type=str,
    default=None,
    help="Path to load the model from. If provided, the model will be loaded from this path.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to save the model. If provided, the model will be saved to this path.",
)

parser.add_argument(
    "--wandb",
    action="store_true",
    help="Enable Weights & Biases logging.",
)
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Name of the Weights & Biases run.",
)
parser.add_argument(
    "--tag",
    type=str,
    default=None,
    help="Tag for the Weights & Biases run.",
)

parser.add_argument(
    "--n_plots",
    type=int,
    default=3,
    help="Number of plots to generate for the results.",
)
