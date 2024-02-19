import os
from time import time
import yaml

from argparse import ArgumentParser, Namespace

from utils.helper_functions import str_time
from nets.segmentationwrapper import SegmentationModel
from trainseg import load_tsdataset

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize
from nets.metrics import *

import pandas as pd
import numpy as np
import torch

import re

def evaluate_model(checkpoint_path, **kwargs):
    print("Testing", checkpoint_path)

    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        **kwargs
    )

    hparam_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path))), "results.dict")

    with open(hparam_path, "r") as f:
        train_args = eval(f.read())
    
    train_args = {**train_args, **kwargs}
    
    dm = load_tsdataset(
        dataset_name=train_args["dataset"],
        dataset_home_directory=train_args["dataset_dir"],
        batch_size=train_args["batch_size"],
        num_workers=train_args["num_workers"],
        window_size=train_args["window_size"],
        window_stride=train_args["window_stride"],
        normalize=train_args["normalize"],
        subjects_for_test=train_args["subjects_for_test"],
        reduce_train_imbalance=train_args["reduce_imbalance"],
        overlap=train_args["overlap"],
        n_val_subjects=train_args["n_val_subjects"]
    )

    data = Trainer().test(datamodule=dm, model=model)

    return model.cm_last

def main(args):
    models_dir = os.listdir(args.model_dir)
    models_dir = list(filter(lambda x: "cache" not in x, models_dir))

    cm = torch.zeros(size=(args.n_cl, args.n_cl), dtype=torch.int64)

    for model_name in models_dir:
        checkpoint_list = os.listdir(os.path.join(args.model_dir, model_name, "version_0", "checkpoints"))

        best_path = None
        best_metric = -float("inf")
        best_epoch = -1e10
        for checkpoint in checkpoint_list:
            match = re.search(r'epoch=([\d.]+)-step=([\d.]+)-val_re=([\d.]+).ckpt', checkpoint)
            if float(match.group(3)) > best_metric:
                best_path = checkpoint
            elif float(match.group(3)) == best_metric and int(match.group(1)) > best_epoch:
                best_path = checkpoint

        print(model_name, best_path)
        cm += evaluate_model(os.path.join(args.model_dir, model_name, "version_0", "checkpoints", best_path), overlap=args.overlap)

    print_cm(cm, args.n_cl)
    for key, value in metrics_from_cm(cm).items():
        print(key, value, "->", value.mean())

    return 0
    

if __name__ == "__main__":
    start_time = time()

    parser = ArgumentParser()

    parser.add_argument("--model_dir", type=str,
        help="Directory with the models")
    parser.add_argument("--track_metric", type=str, default="",
        help="Tracks the following metric to retrieve the best validation checkpoint")
    parser.add_argument("--n_cl", type=int, default=100,
        help="Number of classes")
    parser.add_argument("--overlap", type=int, default=0,
        help="Overlap between observations")

    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")