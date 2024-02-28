from argparse import ArgumentParser
import multiprocessing

def get_parser():
    '''
        dataset, dataset_dir, batch_size, num_workers, window_size, window_stride
        normalize, pattern_size, compute_n, subjects_for_test, encoder_architecture
        decoder_architecture, max_epochs, lr, decoder_features, encoder_features
        decoder_layers, mode, reduce_imbalance, label_mode, num_medoids, voting
        rho, pattern_type, overlap, mtf_bins, training_dir, n_val_subjects, cached
        weight_decayL1, weight_decayL2, pooling
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
    parser.add_argument("--overlap", default=-1, type=int, 
        help="Overlap of observations between training and test examples, default -1 for maximum overlap (equivalent to overlap set to window size -1)")
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

    return parser


def get_model_name(args):
    modelname = f"{args.mode}|{args.dataset}," + '-'.join([str(subject) for subject in args.subjects_for_test]) + f"|{args.n_val_subjects}|" \
                f"{args.window_size},{args.window_stride}|bs{args.batch_size}_lr{args.lr}_l1{args.weight_decayL1}_l2{args.weight_decayL2}|" + \
                f"{args.encoder_architecture}{args.encoder_features}|" + \
                f"{args.decoder_architecture}{args.decoder_features},{args.decoder_layers}|" + \
                (f"m{args.label_mode}|" if args.label_mode > 1 else "") + \
                (f"v{args.voting},{args.rho}|" if args.voting > 1 else "") + \
                (f"ov{args.overlap}|" if args.overlap >= 0 else "")
    
    if args.mode in ["img", "dtw", "dtw_c"]:
        modelname += f"p{args.pattern_size},r{args.rho}|"
    if args.mode == "img":
        modelname += f"med{args.num_medoids},{args.compute_n}|" if args.pattern_type == "med" else f"{args.pattern_type}{args.compute_n}|"
    if args.mode == "mtf":
        modelname += f"bin{args.mtf_bins}|"
    if args.mode == "seg":
        modelname += f"{ '-'.join([str(rate) for rate in args.pooling])}_{args.cf}|ks{args.pattern_size}|"

    if args.mode == "clr3":
        modelname += f"decay{args.cf}|"

    return modelname[:-1]


def get_command(args):
    command = f"--mode {args.mode} --dataset {args.dataset} --lr {args.lr} " + \
                "--subjects_for_test " + ' '.join([str(subject) for subject in args.subjects_for_test]) + " " \
                f"--window_size {args.window_size} --window_stride {args.window_stride} --batch_size {args.batch_size} " + \
                f"--encoder_architecture {args.encoder_architecture} --encoder_features {args.encoder_features} " + \
                f"--decoder_architecture {args.decoder_architecture} --decoder_features {args.decoder_features} --decoder_layers {args.decoder_layers} " + \
                (f"--label_mode {args.label_mode} " if args.label_mode > 1 else "") + \
                (f"--voting {args.voting} --rho {args.rho} " if args.voting > 1 else "") + \
                (f"--overlap {args.overlap} " if args.overlap >= 0 else "")
    
    if args.mode in ["img", "dtw", "dtw_c"]:
        command += f"--pattern_size {args.pattern_size} --rho {args.rho} "
    if args.mode == "img":
        command += f"--compute_n {args.compute_n} "
        command += f"--pattern_type {args.pattern_type} "
        command += f"--num_medoids {args.num_medoids} " if args.pattern_type == "med" else ""
    if args.mode == "mtf":
        command += f"--mtf_bins {args.mtf_bins} "
    if args.mode == "seg":
        command += f"--cf {args.cf} --pattern_size {args.pattern_size} --pooling " + ' '.join([str(subject) for subject in args.pooling]) + " "

    command += f"--num_workers {args.num_workers} --max_epochs {args.max_epochs} --normalize --reduce_imbalance "
    command += f"--training_dir {args.training_dir} --n_val_subjects {args.n_val_subjects} "
    command += f"--cached " if args.cached else ""
    command += f"--weight_decayL1 {args.weight_decayL1} "
    command += f"--weight_decayL2 {args.weight_decayL2} "

    return command
