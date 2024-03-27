from argparse import ArgumentParser
import multiprocessing
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Arguments:
    # pylint: disable=line-too-long invalid-name
    dataset: str = "" # Dataset name for training
    dataset_dir: str = "./datasets" # Directory of the dataset
    batch_size: int = 32
    num_workers: int = multiprocessing.cpu_count() // 2
    window_size: int = 32  # Window size of the dissimilarity frames fed to the classifier
    window_stride: int = 1  # Stride used when extracting windows
    normalize: bool = False  # Whether to normalize the dissimilarity frames and STS
    pattern_size: int = 32  # Size of the pattern for computation of dissimilarity frames (not used)
    compute_n: int = 500  # Number of samples extracted from the STS or Dissimilarity frames to compute medoids and/or means for normalization
    subjects_for_test: Tuple[int] = None  # Subjects reserved for testing and validation
    encoder_architecture: str = "cnn"  # Architecture used for the encoder
    decoder_architecture: str = "mlp"  # Architecture of the decoder, mlp with hidden_layers 0 is equivalent to linear
    max_epochs: int = 10
    lr: float = 1e-3
    decoder_features: int = None  # Number of features on decoder hidden layers, ignored when decoder_layers is 0
    encoder_features: int = None
    decoder_layers: int = 1
    mode: str = "img"  # Mode of training, options: ts for time series as input for the model, img (default) for dissimilarity frames as input, dtw for dtw-layer encoding
    reduce_imbalance: bool = False  # Whether to subsample imbalanced classes
    label_mode: int = 1  # Consider the mode (most common) label out of this number of labels for training (default 1), must be an odd number
    num_medoids: int = 1  # Number of medoids per class to use
    voting: int = 1  # Number of previous predictions to consider in the vote of the next prediction, defaults to 1 (no voting)
    rho: float = 0.1  # Parameter of the online-dtw algorithm the window_size-th root is used as the voting parameter
    pattern_type: str = None  # Type of pattern to use during DM computation
    overlap: int = -1  # Overlap of observations between training and test examples, default -1 for maximum overlap (equivalent to overlap set to window size -1)
    mtf_bins: int = 10  # Number of bins for mtf computation
    training_dir: str = "training"  # Directory of model checkpoints
    n_val_subjects: int = 1  # Number of subjects for validation
    cached: bool = False  # If not set the DF are computed to RAM, patterns are saved regardless
    weight_decayL1: float = 0  # Parameter controlling L1 regularizer
    weight_decayL2: float = 0  # Parameter controlling L2 regularizer
    pooling: Tuple[int] = None  # Pooling in each layer of utime
    cf: float = 1  # Complexity factor of utime
    same_class: bool = False  # Same class windows only for training
    encoder_layers: int = 1  # Layers of encoder
    use_triplets: bool = False  # Return (random) triplets when accessing dataset
    random_seed: int = 42  # Random seed set for RNGs
    pattern_stride: int = 1 # stride of the dissimilarity frames

def get_parser():
    parser = ArgumentParser()

    for argument, typehint in Arguments.__annotations__.items():
        if typehint == bool:
            parser.add_argument(
                f"--{argument}",
                action="store_false" if Arguments.__dict__[argument] else "store_true")
        elif typehint == Tuple[int]:
            parser.add_argument(
                f"--{argument}", nargs="+", type=int, default=Arguments.__dict__[argument])
        else:
            parser.add_argument(
                f"--{argument}", type=typehint, default=Arguments.__dict__[argument])

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
