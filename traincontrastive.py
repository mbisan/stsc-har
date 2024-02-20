import os
from time import time

from utils.helper_functions import load_dataset, str_time
from data.contrastive_dataset import LConDataset
from data.base import split_by_test_subject

from nets.wrapper import ContrastiveWrapper

from pytorch_lightning import seed_everything
import numpy as np

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

    model = ContrastiveWrapper(
        args.arch, dm.n_dims, args.latent_features, args.lr, args.weight_decayL1, args.weight_decayL2, modelname, window_size=args.window_size)
    
    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        })
    
    with open(os.path.join(args.training_dir, modeldir, "results.dict"), "w") as f:
        f.write(str({**data, **args.__dict__, "name": modelname}))
    
    print(data)

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

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(42)
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
