import os
from time import time

str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

from argparse import ArgumentParser
import multiprocessing
import pandas as pd

# dataset imports
from data.har.har_datasets import *
from data.dfdataset import LDFDataset, DFDataset
from data.stsdataset import LSTSDataset
from data.har.label_mappings import *
from data.base import split_by_test_subject

from utils.pattern_extract import *
from utils.methods import create_model_from_DM

import torch
from torchvision.transforms import Normalize

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
    parser.add_argument("--pattern_size", default=32, type=int, 
        help="Size of the pattern for computation of dissimilarity frames (not used)")
    parser.add_argument("--compute_n", default=500, type=int, 
        help="Number of samples extracted from the STS or Dissimilarity frames to compute medoids and/or means for normalization")
    parser.add_argument("--subjects_for_test", nargs="+", type=int, 
        help="Subjects reserved for testing and validation")
    parser.add_argument("--encoder_architecture", default="cnn", type=str, 
        help="Architecture used for the encoder")
    parser.add_argument("--decoder_architecture", default="mlp", type=str,
        help="Architecture of the decoder, mlp with hidden_layers 0 is equivatent to linear")
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--decoder_features", default=None, type=int,
        help="Number of features on decoder hidden layers, ignored when decoder_layers is 0")
    parser.add_argument("--encoder_features", default=None, type=int)
    parser.add_argument("--decoder_layers", default=1, type=int)
    parser.add_argument("--mode", default="img", type=str,
        help="Mode of training, options: ts for time series as input for the model, img (default) for dissimilarity frames as input, dtw for dtw-layer encoding")
    parser.add_argument("--reduce_imbalance", action="store_true", 
        help="Wether to subsample imbalanced classes")
    parser.add_argument("--no-reduce_imbalance", dest="reduce_imbalance", action="store_false")
    parser.add_argument("--label_mode", default=1, type=int,
        help="Consider the mode (most common) label out of this number of labels for training (default 1), must be an odd number")
    parser.add_argument("--num_medoids", default=1, type=int,
        help="Number of medoids per class to use")
    parser.add_argument("--voting", default=1, type=int,
        help="Number of previous predictions to consider in the vote of the next prediction, defaults to 1 (no voting)")
    parser.add_argument("--rho", default=0.1, type=float,
        help="Parameter of the online-dtw algorithm, the window_size-th root is used as the voting parameter")
    parser.add_argument("--pattern_type", type=str, default="med",
        help="Type of pattern to use during DM computation") # pattern types: "med"
    # parser.add_argument("--overlap", default=-1, type=int, 
    #     help="Overlap of observations between training and test examples, default -1 for maximum overlap (equivalent to overlap set to window size -1)")
    parser.add_argument("--mtf_bins", default=10, type=int, 
        help="Number of bins for mtf computation")
    parser.add_argument("--training_dir", default="training", type=str, 
        help="Directory of model checkpoints")
    parser.add_argument("--n_val_subjects", default=1, type=int, 
        help="Number of subjects for validation")
    parser.add_argument("--cached", action="store_true", 
        help="If not set the DF are computed to RAM, patterns are saved regardless")
    parser.add_argument("--weight_decayL1", default=0, type=float,
        help="Parameter controlling L1 regularizer")
    parser.add_argument("--weight_decayL2", default=0, type=float,
        help="Parameter controlling L2 regularizer")

    return parser

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
        overlap = -1,
        n_val_subjects = 1,
        cached = True,
        patterns = None):

    actual_window_size = window_size
    if pattern_size > window_size:
        window_size = pattern_size

    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
    ds.wstride = 1 # window stride to compute medoids must be 1

    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {ds.wsize}")

    if patterns is None:
        if pattern_type == "med":
            print("Computing medoids...")
            meds = sts_medoids(ds, pattern_size=pattern_size, meds_per_class=num_medoids, n=compute_n)
        elif pattern_type == "syn":
            print("Using synthetic shapes...")
            meds = np.empty((3, pattern_size))
            meds[0,:] = np.linspace(-1, 1, pattern_size)
            meds[1,:] = np.linspace(1, -1, pattern_size)
            meds[2,:] = 0
        elif pattern_type == "syn_1":
            print("Using up shape...")
            meds = np.empty((1, pattern_size))
            meds[0,:] = np.linspace(-1, 1, pattern_size)
        elif pattern_type == "syn_2":
            print("Using up down shape...")
            meds = np.empty((2, pattern_size))
            meds[0,:] = np.linspace(-1, 1, pattern_size)
            meds[1,:] = np.linspace(1, -1, pattern_size)
        elif pattern_type == "syn_g":
            print("Using synthetic shapes with gaussian noise...")
            meds = np.empty((3, pattern_size))
            meds[0,:] = np.linspace(-1, 1, pattern_size) + 0.2 * np.random.randn(pattern_size) # sigma = 0.2*0.2 = 0.04 (std)
            meds[1,:] = np.linspace(1, -1, pattern_size) + 0.2 * np.random.randn(pattern_size)
            meds[2,:] = 0.1 * np.random.randn(pattern_size)
        elif pattern_type == "freq":
            print("Using sinusoidal with predominant frequencies...")
            fft_mag = process_fft(ds.STS, ds.SCS)
            pred_freq = get_predominant_frequency(fft_mag, mode="per_class")
            meds = np.empty((pred_freq.shape[0], pred_freq.shape[1], pattern_size))
            for i in range(pred_freq.shape[0]):
                for j in range(pred_freq.shape[1]):
                    meds[i, j, :] = np.sin(2*np.pi* pred_freq[i, j] *np.arange(pattern_size))
        elif pattern_type == "freq_c":
            print("Using sinusoidal with predominant frequencies per channel...")
            fft_mag = process_fft(ds.STS, ds.SCS)
            pred_freq = get_predominant_frequency(fft_mag, mode="per_channel") # shape (freqs, 2) i,e, value, magnitude
            NUM_WAVES = 3
            meds = np.empty((NUM_WAVES, pattern_size))
            for i in range(NUM_WAVES):
                meds[i, :] = np.sin(2*np.pi* pred_freq[i, 0] *np.arange(pattern_size))
        elif pattern_type == "f1":
            meds = np.empty((1, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size) # one cycle
        elif pattern_type == "f2":
            meds = np.empty((1, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size) # two cycle
        elif pattern_type == "f4":
            meds = np.empty((1, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size) # four cycle
        elif pattern_type == "f12":
            meds = np.empty((2, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
            meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
        elif pattern_type == "f14":
            meds = np.empty((2, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
            meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
        elif pattern_type == "f24":
            meds = np.empty((2, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
            meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
        elif pattern_type == "f124":
            meds = np.empty((3, pattern_size))
            meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
            meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
            meds[2,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
        elif pattern_type == "fftcoef":
            print("Using fft coefficients for the pattern...")
            pattern_freq = np.fft.fftfreq(pattern_size)[:pattern_size//2]
            fft_coef = process_fft_frequencies(ds.STS, ds.SCS, pattern_freq)

            if 100 in fft_coef.keys():
                del fft_coef[100] # remove the ignore label

            meds = np.zeros((len(fft_coef.keys()), ds.STS.shape[0], pattern_size)) # num_classes, channel, pattern_size
            for i, coef in enumerate(fft_coef.values()):
                for c in range(meds.shape[1]):
                    for j, m in enumerate(pattern_freq):
                        meds[i, c, :] += coef[c, j].real * np.sin(2*np.pi* m * np.arange(pattern_size))
                        meds[i, c, :] += coef[c, j].imag * np.cos(2*np.pi* m * np.arange(pattern_size))
        elif pattern_type == "fftvar":
            print("Using fft coefficient variances across classes for the pattern...")
            pattern_freq = np.fft.fftfreq(pattern_size)[:pattern_size//2]
            fft_coef = process_fft_frequencies(ds.STS, ds.SCS, pattern_freq)

            if 100 in fft_coef.keys():
                del fft_coef[100] # remove the ignore label

            # compute variance across classes
            fft_coef_all = np.zeros((len(fft_coef.keys()), ds.STS.shape[0], pattern_freq.shape[0]), dtype=np.complex128)
            for i, v in enumerate(fft_coef.values()):
                fft_coef_all[i, :, :] = v

            fft_coef_mean = np.mean(fft_coef_all, axis=(0, 1))

            fft_var = np.var(np.abs(fft_coef_all), axis=(0, 1)) # shape of pattern_freq
            fft_freq_ordered = pattern_freq[np.argsort(fft_var)] # from lower to higher variance
            fft_coef_mean_sorted = fft_coef_mean[np.argsort(fft_var)] # from lower to higher variance

            NUM_WAVES = 3
            meds = np.zeros((NUM_WAVES, pattern_size)) # num_classes, channel, pattern_size
            for i in range(NUM_WAVES):
                meds[i, :] += fft_coef_mean_sorted[-i].real * np.sin(2*np.pi* fft_freq_ordered[-i] * np.arange(pattern_size))
                meds[i, :] += fft_coef_mean_sorted[-i].imag * np.cos(2*np.pi* fft_freq_ordered[-i] * np.arange(pattern_size))

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
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, label_mode=label_mode, overlap=overlap)

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
        overlap = -1,
        mode = None,
        mtf_bins = 50,
        n_val_subjects = 1):
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    data_split = split_by_test_subject(ds, subjects_for_test, n_val_subjects)

    dm = LSTSDataset(ds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, label_mode=label_mode, overlap=overlap, mode=mode, mtf_bins=mtf_bins)
    dm.l_patterns = pattern_size

    return dm

def load_dm(args, patterns = None):
    if args.mode == "img":
        dm = load_dmdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            rho=args.rho, batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size, 
            compute_n=args.compute_n, subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, num_medoids=args.num_medoids, pattern_type=args.pattern_type, overlap=args.overlap, 
            n_val_subjects=args.n_val_subjects, cached=args.cached, patterns=patterns)
    elif args.mode in ["ts", "dtw", "dtw_c", "mtf", "gasf", "gadf", "fft", "cwt_test", "dtwfeats"]:
        dm = load_tsdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size,
            subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, overlap=args.overlap, mode=args.mode, mtf_bins=args.mtf_bins, n_val_subjects=args.n_val_subjects)

    print(f"Using {len(dm.ds_train)} observations for training, {len(dm.ds_val)} for validation and {len(dm.ds_test)} observations for test")
  
    return dm

def get_model(name, args, dm):

    model = create_model_from_DM(dm, name=name, 
        dsrc=args.mode, arch=args.encoder_architecture, dec_arch=args.decoder_architecture,
        task="cls", lr=args.lr, enc_feats=args.encoder_features, 
        dec_feats=args.decoder_features, dec_layers=args.decoder_layers,
        voting={"n": args.voting, "rho": args.rho}, weight_decayL1=args.weight_decayL1, weight_decayL2=args.weight_decayL2, args=str(args.__dict__))
    
    return model
