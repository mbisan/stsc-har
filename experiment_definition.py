cross_validate = {
    "HARTH": [[21],[20],
              [19],[18],[17],[16],[15],[14],[13],[12],[11],[10],
              [9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "UCI-HAR": [[29],[28],[27],[26],[25],[24],[23],[22],[21],[20],
                [19],[18],[17],[16],[15],[14],[13],[12],[11],[10],
                [9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "WISDM": [[35],[34],[33],[32],[31],[30],
              [29],[28],[27],[26],[25],[24],[23],[22],[21],[20],
              [19],[18],[17],[16],[15],[14],[13],[12],[11],[10],
              [9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "HARTH_g": [[21],[20],
                [19],[18],[17],[16],[15],[14],[13],[12],[11],[10],
                [9],[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "PAMAP2": [[8],[7],[6],[5],[4],[3],[2],[1],[0]],
    "MHEALTH": [[9],[8],[7],[6],[5],[4],[3],[2],[1],[0]]
}

baseArguments = {
    "num_workers": 8,
    "lr": 0.001,
    "n_val_subjects": 4,
    "batch_size": 128,
    "label_mode": 1,
    "voting": 1,
    "overlap": -1,
    "weight_decayL1": 0.0001,
    "weight_decayL2": 0.00001,
    "command": "training.py",
    "cached": False,
    "ram": 8,
    "cpus": 8,
    "rho": 0,
    "reduce_imbalance": True,
    "normalize": True
}

harth_ts = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 50,
    "window_stride": 1,
    "mode": "ts",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 10,
    "training_dir": "_harth_ts",
}

uci_ts = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 50,
    "window_stride": 1,
    "mode": "ts",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_uci_ts",
}

wisdm_ts = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 20,
    "window_stride": 1,
    "mode": "ts",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_wisdm_ts",
}

harth_img = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 25,
    "window_stride": 2,
    "mode": "img",
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 10,
    "training_dir": "_harth_img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": ["med", "syn", "syn_2", "fftcoef", "fftvar"],
    "pattern_size": 25,
    "cached": False,
    "rho": 0.1,
    "ram": 24
}

harth_img_g = {
    "dataset": "HARTH_g",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 25,
    "window_stride": 2,
    "mode": "img",
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 10,
    "training_dir": "_harth_img_g",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": ["med", "fftcoef"],
    "pattern_size": 25,
    "cached": False,
    "rho": 0.1,
    "ram": 24
}

uci_img = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 25,
    "window_stride": 2,
    "mode": "img",
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_uci_img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": ["med", "syn", "syn_2", "fftcoef", "fftvar", "noise"],
    "pattern_size": 25,
    "cached": False,
    "rho": 0.1,
}

wisdm_img = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 16,
    "window_stride": 2,
    "mode": "img",
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_wisdm_img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": ["med", "syn", "syn_2", "fftcoef", "fftvar"],
    "pattern_size": 16,
    "cached": False,
    "rho": 0.1,
}

harth_tr = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 50,
    "window_stride": 1,
    "mode": ["gasf", "gadf", "mtf"],
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 10,
    "training_dir": "_harth_tr",
    "mtf_bins": 16,
}

uci_tr = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 50,
    "window_stride": 1,
    "mode": ["gasf", "gadf", "mtf"],
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_uci_tr",
    "mtf_bins": 16,
}

wisdm_tr = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 20,
    "window_stride": 1,
    "mode": ["gasf", "gadf", "mtf"],
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_wisdm_tr",
    "mtf_bins": 8,
}

harth_seg = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 120,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 12,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 10,
    "training_dir": "_harth_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
    "batch_size": 64
}

uci_seg = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 120,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 10,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 30,
    "training_dir": "_uci_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
    "batch_size": 64
}

wisdm_seg = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 48,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 16,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 30,
    "training_dir": "_wisdm_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
    "batch_size": 64
}

uci_clr = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 40,
    "window_stride": 1,
    "mode": "clr",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 10,
    "decoder_architecture": "cnn_ts_dec", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 30,
    "training_dir": "_uci_clr_cpmodel",
    "cf": 0,
}

uci_seg_class = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 50,
    "window_stride": 1,
    "mode": "ts",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_uci_seg_class",
    "same_class": True
}

experiments = [uci_clr, uci_seg_class]
