from argparse import ArgumentParser
import multiprocessing

def get_parser():
    '''
        dataset, dataset_dir, batch_size, num_workers, window_size, window_stride
        normalize, pattern_size, compute_n, subjects_for_test, encoder_architecture
        decoder_architecture, max_epochs, lr, decoder_features, encoder_features
        decoder_layers, mode, reduce_imbalance, label_mode, num_medoids, voting
        rho, pattern_type, overlap, mtf_bins, training_dir, n_val_subjects, cached
        weight_decayL1, weight_decayL2, pooling, cf, encoder_layers, use_triplets
    '''
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
        help="""Number of samples extracted from the STS or Dissimilarity frames to
        compute medoids and/or means for normalization""")
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
        help="""Mode of training, options: ts for time series as input for the model,
        img (default) for dissimilarity frames as input, dtw for dtw-layer encoding""")
    parser.add_argument("--reduce_imbalance", action="store_true",
        help="Wether to subsample imbalanced classes")
    parser.add_argument("--no-reduce_imbalance", dest="reduce_imbalance", action="store_false")
    parser.add_argument("--label_mode", default=1, type=int,
        help="""Consider the mode (most common) label out of this number of labels
        for training (default 1), must be an odd number""")
    parser.add_argument("--num_medoids", default=1, type=int,
        help="Number of medoids per class to use")
    parser.add_argument("--voting", default=1, type=int,
        help="""Number of previous predictions to consider in the vote of the next prediction
        defaults to 1 (no voting)""")
    parser.add_argument("--rho", default=0.1, type=float,
        help="""Parameter of the online-dtw algorithm
        the window_size-th root is used as the voting parameter""")
    parser.add_argument("--pattern_type", type=str, default=None,
        help="Type of pattern to use during DM computation") # pattern types: "med"
    parser.add_argument("--overlap", default=-1, type=int,
        help="""Overlap of observations between training and test examples
        default -1 for maximum overlap (equivalent to overlap set to window size -1)""")
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
    parser.add_argument("--pooling", nargs="+", type=int, default=None,
        help="Pooling in each layer of utime")
    parser.add_argument("--cf", default=1, type=float,
        help="Complexity factor of utime")
    parser.add_argument("--same_class", action="store_true",
        help="Same class windows only for training")
    parser.add_argument("--encoder_layers", type=int, default=1,
        help="Layers of encoder")
    parser.add_argument("--use_triplets", action="store_true",
        help="Return (random) triplets when accessing dataset")
    parser.add_argument("--random_seed", type=int, default=42,
        help="Random seed set for RNGs")

    return parser


def get_model_name(args):
    modelname = ""

    for element, value in args.__dict__.items():
        if element in ["command", "ram", "cpus", "dataset_dir", "training_dir"]:
            continue
        if isinstance(value, bool):
            if value:
                modelname += f"{element[0]}"
        elif isinstance(value, list):
            modelname += f"{element[0]}"
            modelname += '-'.join([str(val) for val in value])
        elif isinstance(value, int):
            modelname += f"{element[0]}{value}"
        elif isinstance(value, str):
            modelname += f"{value}"
        elif isinstance(value, float):
            modelname += f"{element[0]}{value:.1E}"

    return modelname


def get_command(args):
    command = ""

    for element, value in args.__dict__.items():
        if element in ["command", "ram", "cpus"]:
            continue
        if isinstance(value, bool):
            if value:
                command += f"--{element} "
        elif isinstance(value, list):
            command += f"--{element} "
            command += ' '.join([str(val) for val in value]) + " "
        else:
            command += f"--{element} {value} "

    return command
