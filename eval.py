import os
from time import time
import yaml

from argparse import ArgumentParser, Namespace

from nets.wrapper import WrapperModel
from utils.helper_functions import load_dm, str_time

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize

import pandas as pd
import numpy as np

def main(args):
    hparam_path = os.path.join(args.model_dir, "hparams.yaml")
    train_args = None

    with open(hparam_path, "r") as f:
        try:
            train_args = Namespace(**eval(yaml.safe_load(f)["args"]))
        except yaml.YAMLError as exc:
            print(exc)
            return 1
    
    trainer = Trainer()
    patterns = np.load(os.path.join(os.path.dirname(args.model_dir), "pattern.npz"))

    dm = load_dm(train_args, patterns=patterns)

    table = pd.DataFrame(columns=["name"])

    best_checkpoint = None if args.just_test == "" else args.just_test
    best_metric = -float("inf")

    if args.just_test == "":
        for checkpoint_file in sorted(os.listdir(os.path.join(args.model_dir, "checkpoints"))):
            print("Validating", os.path.join(args.model_dir, "checkpoints", checkpoint_file))
            model = WrapperModel.load_from_checkpoint(
                checkpoint_path=os.path.join(args.model_dir, "checkpoints", checkpoint_file),
                hparams_file=hparam_path,
                voting={"n": args.voting, "rho": train_args.rho}
            )
            # print(summarize(model, 1))
            data = trainer.validate(model=model, datamodule=dm)
            print(data[0])

            if data[0][args.track_metric] > best_metric:
                best_metric = data[0][args.track_metric]
                best_checkpoint = checkpoint_file

            for key in data[0]:
                if key not in table.columns:
                    table[key] = None
            table.loc[len(table)] = {"name": checkpoint_file, **data[0]}

    # test the best checkpoint
    print("------------ Testing the best model ------------")
    model = WrapperModel.load_from_checkpoint(
        checkpoint_path=os.path.join(args.model_dir, "checkpoints", best_checkpoint),
        hparams_file=hparam_path,
        voting={"n": args.voting, "rho": train_args.rho}
    )
    print(summarize(model, 1))
    data = trainer.test(model=model, datamodule=dm)
    print(data[0])

    for key in data[0]:
        if key not in table.columns:
            table[key] = None
    table.loc[len(table)] = {"name": best_checkpoint, **data[0]}

    print(table)
    table.to_csv(os.path.join(args.model_dir, "evaluation_test_" + args.track_metric + ".csv"))

    return 0
    

if __name__ == "__main__":
    start_time = time()

    parser = ArgumentParser()

    parser.add_argument("--model_dir", type=str,
        help="Model directory")
    parser.add_argument("--track_metric", type=str, default="",
        help="Tracks the following metric to retrieve the best validation checkpoint")
    parser.add_argument("--voting", type=int, default=1,
        help="Number of observations to use while voting the prediction")
    parser.add_argument("--just_test", type=str, default="",
        help="If this argument is nonempty, only the checkpoint .ckpt is loaded and tested")

    args = parser.parse_args()

    if args.just_test != "":
        args.model_dir = os.path.dirname(os.path.dirname(args.just_test))
        args.just_test = os.path.split(args.just_test)[-1]
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")