from torch.utils.tensorboard import SummaryWriter # This hack avoids segfault
import argparse
import faulthandler 
faulthandler.enable()

from hotels50k import UseOriginalTestSplitManager, Hotels50kDataset
import os

cwd = os.getcwd()

def get_runner():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pytorch_home", type=str, default="~/.cache/torch/") # absolute path
    parser.add_argument("--dataset_root", type=str, default=f"{cwd}/datasets") # absolute path
    parser.add_argument("--root_experiment_folder", type=str, default=f"{cwd}/experiments") # absolute path
    parser.add_argument("--global_db_path", type=str, default=None)
    parser.add_argument("--merge_argparse_when_resuming", default=False, action='store_true')
    parser.add_argument("--root_config_folder", type=str, default=None)
    parser.add_argument("--bayes_opt_iters", type=int, default=10)
    parser.add_argument("--reproductions", type=str, default="5")
    args, _ = parser.parse_known_args()

    if args.bayes_opt_iters > 0:
        from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner
        args.reproductions = [int(x) for x in args.reproductions.split(",")]
        runner = BayesOptRunner
    else:
        from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner
        runner = SingleExperimentRunner
        del args.bayes_opt_iters
        del args.reproductions

    r = runner(**(args.__dict__))
    r.register('dataset', Hotels50kDataset)
    r.register('split_manager', UseOriginalTestSplitManager)
    return r
