import os
from time import time

str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

# dataset imports
from data.har.har_datasets import *
from data.dfdataset import LDFDataset, DFDataset
from data.stsdataset import LSTSDataset
from data.har.label_mappings import *
from data.base import split_by_test_subject

from utils.patterns import get_patterns

import torch
from torchvision.transforms import Normalize

def cm_str(cm):
    if cm is None:
        return ""
    str_res = "<>[\n"
    for i in range(len(cm)):
        str_res += "    [" + ",".join([f"{cm[i][j]:>6}" for j in range(len(cm[i]))]) + "],\n"
    str_res = str_res[:-2] + "\n  <>]"
    return str_res

def load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize):
 
    if dataset_home_directory is None:
        dataset_home_directory = "./datasets"

    ds = None
    if dataset_name == "WISDM":
        ds = WISDMDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "UCI-HAR":
        ds = UCI_HARDataset(
            os.path.join(dataset_home_directory, dataset_name),
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=ucihar_label_mapping)
    elif dataset_name == "HARTH":
        ds = HARTHDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
    elif dataset_name == "HARTH_g":
        ds = HARTHDataset(
            os.path.join(dataset_home_directory, "HARTH"), 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
        ds.feature_group = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    
    return ds

def load_dmdataset(
        dataset_name,
        dataset_home_directory = None,
        rho = 0.1,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        compute_n = 500,
        subjects_for_test = None,
        reduce_train_imbalance = False,
        num_medoids = 1,
        label_mode = 1,
        pattern_type = "med",
        # overlap = -1,
        n_val_subjects = 1,
        cached = True,
        patterns = None,
        same_class = False):

    actual_window_size = window_size
    if pattern_size > window_size:
        window_size = pattern_size

    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
    ds.wstride = 1 # window stride to compute medoids must be 1

    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {ds.wsize}")

    if patterns is None:
        meds = get_patterns(pattern_type, pattern_size, num_medoids, compute_n, ds)
    else:
        meds=patterns

    if pattern_size > actual_window_size:
        ds.wsize = actual_window_size

    ds.wstride = window_stride # restore original wstride

    print("Computing dissimilarity frames...")
    dfds = DFDataset(ds, patterns=meds, rho=rho, dm_transform=None, cached=cached, dataset_name=dataset_name)

    data_split = split_by_test_subject(ds, subjects_for_test, n_val_subjects)

    if normalize:
        # get average values of the DM
        DM = []
        np.random.seed(42)
        for i in np.random.choice(np.arange(len(dfds))[data_split["train"](dfds.stsds.indices)], compute_n):
            dm, _, _ = dfds[i]
            DM.append(dm)
        DM = torch.stack(DM)

        dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))
        dfds.dm_transform = dm_transform

    dm = LDFDataset(dfds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, label_mode=label_mode,
        same_class=same_class) #, overlap=overlap)

    return dm


def load_tsdataset(
        dataset_name,
        dataset_home_directory = None,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        subjects_for_test = None,
        reduce_train_imbalance = False,
        label_mode = 1,
        # overlap = -1,
        mode = None,
        mtf_bins = 50,
        n_val_subjects = 1,
        skip = 1,
        same_class = False):
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    data_split = split_by_test_subject(ds, subjects_for_test, n_val_subjects)

    dm = LSTSDataset(ds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, 
        label_mode=label_mode, mode=mode, mtf_bins=mtf_bins, skip=skip, same_class=same_class) # overlap=overlap, 
    dm.l_patterns = pattern_size

    return dm


def load_dm(args, patterns = None):
    if args.mode == "img":
        dm = load_dmdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            rho=args.rho, batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size, 
            compute_n=args.compute_n, subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, num_medoids=args.num_medoids, pattern_type=args.pattern_type, # overlap=args.overlap, 
            n_val_subjects=args.n_val_subjects, cached=args.cached, patterns=patterns, same_class=args.same_class)
    else:
        dm = load_tsdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size,
            subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, mode=args.mode, mtf_bins=args.mtf_bins, n_val_subjects=args.n_val_subjects,
            skip=(args.window_size - args.overlap) if args.overlap>=0 else 1, same_class=args.same_class) 

    print(f"Using {len(dm.ds_train)} observations for training, {len(dm.ds_val)} for validation and {len(dm.ds_test)} observations for test")
  
    return dm
