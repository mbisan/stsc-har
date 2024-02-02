baseArguments = {
    "num_workers": 8,
    "dataset": "HARTH",
    "subjects_for_test": [
        [21],
        [20],
        [19],
        [18],
        [17],
        [16],
        [15],
        [14],
        [13],
        [12],
        [11],
        [10],
        [9],
        [8],
        [7],
        [6],
        [5],
        [4],
        [3],
        [2],
        [1],
        [0]
    ],
    "lr": 0.001,
    "n_val_subjects": 4,
    "encoder_architecture": "cnn_gap",
    "encoder_features": 32,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "batch_size": 128,
    "label_mode": 1,
    "voting": 1,
    "rho": 0.1,
    # "overlap": -1,
    "max_epochs": 20,
    "training_dir": "training_syn",
    "cached": False,
    "weight_decayL1": 0.0001,
    "weight_decayL2": 0.00001
}

imgExperiments = {
    "window_size": 50,
    "window_stride": 2,
    "mode": "img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": "syn",
    "pattern_size": 50
}

dtwExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw",
    "pattern_size": [8, 16, 24]
}

dtwcExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw_c",
    "pattern_size": [8, 16, 24]
}

tsExperiments = {
    "window_size": 50,
    "window_stride": 1,
    "mode": "ts"
}

gasfExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "gasf"
}

gadfExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "gadf"
}

mtffExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "mtf",
    "mtf_bins": 10
}

RAM = 32
CPUS = 16

experiments = [imgExperiments] #, dtwExperiments, dtwcExperiments, tsExperiments, gasfExperiments, gadfExperiments, mtffExperiments]
