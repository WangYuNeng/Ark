import json
import pickle
from argparse import ArgumentParser

from dse import evaluate_puf_single_bit_flip, setup_puf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-p", "--check_point", type=str, required=True)
    parser.add_argument("--n_inst", type=int, default=None)
    parser.add_argument("--center_chls_size", type=int, default=None)
    parser.add_argument("--n_core", type=int, default=None)
    parser.add_argument("-s", "--save_crps", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    config["checkpoint"] = args.check_point
    params = pickle.load(open(args.check_point, "rb"))
    print("Loaded checkpoint from {}".format(args.check_point))
    print("Params: {}".format(params))

    if args.save_crps is not None:
        if args.n_inst is not None:
            config["n_inst"] = args.n_inst
        if args.center_chls_size is not None:
            config["center_chls_size"] = args.center_chls_size
        if args.n_core is not None:
            config["n_core"] = args.n_core

        puf, init_sol = setup_puf(config)

        crps = evaluate_puf_single_bit_flip(
            params=params,
            puf=puf,
            center_chls_size=config["center_chls_size"],
            window_size=config["window_size"],
            n_inst=config["n_inst"],
            n_core=config["n_core"],
            plot=False,
            return_crps=True,
            tqdm_process=True,
        )
        if args.save_crps is not None:
            with open(args.save_crps, "wb") as f:
                pickle.dump(crps, f)
