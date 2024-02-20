cross_validate = {
    "HARTH": [[21],[20],[19],[18],[17],[16],[15],[14],[13],[12],[11],[10],[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "UCI-HAR": [[29],[28],[27],[26],[25],[24],[23],[22],[21],[20],[19],[18],[17],[16],[15],[14],[13],[12],[11],[10],[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "WISDM": [[35],[34],[33],[32],[31],[30],[29],[28],[27],[26],[25],[24],[23],[22],[21],[20],[19],[18],[17],[16],[15],[14],[13],[12],[11],[10],[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "HARTH_g": [[21],[20],[19],[18],[17],[16],[15],[14],[13],[12],[11],[10],[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],    
}

dataset = "UCI-HAR"

baseArguments = {
    "num_workers": 8,
    "dataset": dataset,
    "subjects_for_test": [[1, 2, 3]], #cross_validate[dataset],
    "lr": 0.001,
    "n_val_subjects": 4,
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "batch_size": 128,
    "label_mode": 1,
    "voting": 1,
    "overlap": -1,
    "max_epochs": 20,
    "training_dir": "training_tests3",
    "weight_decayL1": 0.0001,
    "weight_decayL2": 0.00001,
    "command": "training.py",
    "cached": False
}

imgExperiments = {
    "window_size": 25,
    "window_stride": 2,
    "mode": "img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": "syn_2",
    "pattern_size": 25,
    "cached": False,
    "rho": 0.1,
}

dtwExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw",
    "pattern_size": [8, 16, 24],
}

dtwcExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw_c",
    "pattern_size": [8, 16, 24],
}

tsExperiments = {
    "window_size": 50,
    "window_stride": 1,
    "mode": "ts",
}

gasfExperiments = {
    "window_size": 50,
    "window_stride": 1,
    "mode": "gasf",
}

gadfExperiments = {
    "window_size": 40,
    "window_stride": 1,
    "mode": "gadf",
}

mtffExperiments = {
    "window_size": 40,
    "window_stride": 1,
    "mode": "mtf",
    "mtf_bins": 16,
}

segExperiments = {
    "mode": "seg",
    "window_size": 128,
    "window_stride": 1,
    "pooling": [[2, 2, 2]],
    "cf": 1.5,
    "pattern_size": 5
}

RAM = 16
CPUS = 16

experiments = [imgExperiments, segExperiments] #, dtwExperiments, dtwcExperiments, tsExperiments, gasfExperiments, gadfExperiments, mtffExperiments]
