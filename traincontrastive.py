import os
from time import time

from utils.helper_functions import load_dataset, str_time
from data.contrastive_dataset import LConDataset
from data.base import split_by_test_subject

from nets.contrastivewrapper import ContrastiveModel

from pytorch_lightning import seed_everything
import numpy as np

from argparse import ArgumentParser
import multiprocessing

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

import warnings # shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def main(args):
    dm = load_tsdataset(dataset_name=args.dataset, dataset_home_directory=args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, subjects_for_test=args.subjects_for_test,
        n_val_subjects=args.n_val_subjects)
    print(f"Using {len(dm.ds_train)} observations for training, {len(dm.ds_val)} for validation and {len(dm.ds_test)} observations for test")

    modelname = get_model_name(args)
    print("\n" + modelname)
    modeldir = modelname.replace("|", "_").replace(",", "_")

    model = ContrastiveModel(
        args.arch, dm.n_dims, args.latent_features, args.lr, args.weight_decayL1, args.weight_decayL2, modelname)

    # save computed patterns for later use
    if not os.path.exists(os.path.join(args.training_dir)):
        os.mkdir(os.path.join(args.training_dir))    
    if not os.path.exists(os.path.join(args.training_dir, modeldir)):
        os.mkdir(os.path.join(args.training_dir, modeldir))
    
    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        })
    
    with open(os.path.join(args.training_dir, modeldir, "results.dict"), "w") as f:
        f.write(str({**data, **args.__dict__, "name": modelname}))
    
    print(data)

def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str,
        help="Dataset name for training")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, 
        help="Directory of the dataset")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count()//2, type=int)
    parser.add_argument("--window_size", default=32, type=int, 
        help="Window size of the dissimilarity frames fed to the classifier")
    parser.add_argument("--window_stride", default=1, type=int, 
        help="Stride used when extracting windows")
    parser.add_argument("--normalize", action="store_true", 
        help="Wether to normalize the dissimilarity frames and STS")
    parser.add_argument("--subjects_for_test", nargs="+", type=int, 
        help="Subjects reserved for testing and validation")
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--training_dir", default="training", type=str, 
        help="Directory of model checkpoints")
    parser.add_argument("--n_val_subjects", default=1, type=int, 
        help="Number of subjects for validation")
    parser.add_argument("--weight_decayL1", default=0, type=float,
        help="Parameter controlling L1 regularizer")
    parser.add_argument("--weight_decayL2", default=0, type=float,
        help="Parameter controlling L2 regularizer")
    parser.add_argument("--latent_features", default=10, type=int,
        help="Number of latent features")
    parser.add_argument("--arch", type=str,
        help="Architecture of segmentation model")

    return parser

def get_model_name(args):
    modelname = f"con|{args.dataset}," + '-'.join([str(subject) for subject in args.subjects_for_test]) + f"|{args.n_val_subjects}|" \
                f"{args.window_size},{args.window_stride}|bs{args.batch_size}_lr{args.lr}_l1{args.weight_decayL1}_l2{args.weight_decayL2}|" + \
                f"{args.latent_features}|" + \
                f"{args.arch}|"

    return modelname[:-1]

def get_command(args):
    command = f"--dataset {args.dataset} --lr {args.lr} " + \
                "--subjects_for_test " + ' '.join([str(subject) for subject in args.subjects_for_test]) + " " \
                f"--window_size {args.window_size} --window_stride {args.window_stride} --batch_size {args.batch_size} " + \
                f"--arch {args.arch} --latent_features {args.latent_features} "

    command += f"--num_workers {args.num_workers} --max_epochs {args.max_epochs} "
    command += f"--training_dir {args.training_dir} --n_val_subjects {args.n_val_subjects} "
    command += f"--weight_decayL1 {args.weight_decayL1} "
    command += f"--weight_decayL2 {args.weight_decayL2} "

    return command


def load_tsdataset(
        dataset_name,
        dataset_home_directory = None,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        subjects_for_test = None,
        n_val_subjects = 1):
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    data_split = split_by_test_subject(ds, subjects_for_test, n_val_subjects)

    dm = LConDataset(ds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers)
    dm.l_patterns = 1

    return dm


def train_model(
        dm, 
        model: ContrastiveModel,
        max_epochs: int,
        pl_kwargs: dict = {"default_root_dir": "training", "accelerator": "auto", "seed": 42}) -> tuple[ContrastiveModel, dict]:
    
    # reset the random seed
    seed_everything(pl_kwargs["seed"], workers=True)

    # choose metrics
    metrics = {"all": ["re", "f1", "auroc"], "target": "val_loss", "mode": "max"}

    # set up the trainer
    ckpt = ModelCheckpoint(
        monitor=metrics['target'], 
        mode=metrics["mode"],
        save_top_k=-1,
        filename='{epoch}-{step}-{val_re:.4f}'
    )

    tr = Trainer(
        default_root_dir=pl_kwargs["default_root_dir"],
        accelerator=pl_kwargs["accelerator"], 
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"), TQDMProgressBar(refresh_rate=25)], 
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir=pl_kwargs["default_root_dir"], name=model.name.replace("|", "_").replace(",", "_"))
    )

    # train the model
    tr.fit(model=model, datamodule=dm)

    # load the best weights
    print(ckpt.best_model_path)
    model = ContrastiveModel.load_from_checkpoint(ckpt.best_model_path)

    # run the validation with the final weights
    data = tr.test(model, datamodule=dm)

    return model, {**data[0], "cm": repr(model.cm_last.tolist()), "path": ckpt.best_model_path}

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(42)
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
