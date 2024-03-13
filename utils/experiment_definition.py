cross_validate = {
    "HARTH": [[i] for i in range(22)],
    "UCI-HAR": [[i] for i in range(30)],
    "WISDM": [[i] for i in range(36)],
    "PAMAP2": [[i] for i in range(9)],
    "MHEALTH": [[i] for i in range(10)],
}

baseArguments = {
    "num_workers": 8,
    "lr": 0.001,
    "n_val_subjects": 3,
    "batch_size": 64,
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
    "normalize": True,
    "max_epochs": 20,
    "same_class": True
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
    "window_size": 25,
    "window_stride": 2,
    "mode": "img",
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "training_dir": "_wisdm_img",
    "num_medoids": 1,
    "compute_n": 300,
    "pattern_type": ["med", "syn", "syn_2", "fftcoef", "fftvar"],
    "pattern_size": 25,
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
    "training_dir": "_uci_tr",
    "mtf_bins": 16,
}

wisdm_tr = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 40,
    "window_stride": 1,
    "mode": ["gasf", "gadf", "mtf"],
    "encoder_architecture": "cnn_gap_img",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "training_dir": "_wisdm_tr",
    "mtf_bins": 8,
}

harth_seg = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 128,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 12,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "training_dir": "_harth_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
}

uci_seg = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 128,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 12,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "training_dir": "_uci_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
}

wisdm_seg = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 80,
    "window_stride": 1,
    "mode": "seg",
    "encoder_architecture": "utime",
    "encoder_features": 12,
    "decoder_architecture": "-", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "training_dir": "_wisdm_seg",
    "cf": 1.5,
    "pattern_size": 3,
    "pooling": [[2, 2, 2]],
    "overlap": 0,
}

uci_clr = {
    "dataset": "UCI-HAR",
    "subjects_for_test": cross_validate["UCI-HAR"],
    "window_size": 40,
    "window_stride": 1,
    "mode": "clr",
    "encoder_architecture": "cnn_ts",
    "encoder_features": 10,
    "decoder_architecture": "cnn_ts_dec", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 30,
    "training_dir": "_uci_clr_cpmodel",
    "cf": 0,
    "same_class": True
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

harth_clr = {
    "dataset": "HARTH",
    "subjects_for_test": cross_validate["HARTH"],
    "window_size": 40,
    "window_stride": 1,
    "mode": "clr",
    "encoder_architecture": "cnn_ts",
    "encoder_features": 10,
    "decoder_architecture": "cnn_ts_dec", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 10,
    "training_dir": "_harth_clr_cpmodel",
    "cf": 0,
    "same_class": True
}

harth_seg_class = {
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
    "training_dir": "_harth_seg_class",
    "same_class": True
}

wisdm_clr = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 20,
    "window_stride": 1,
    "mode": "clr",
    "encoder_architecture": "cnn_ts",
    "encoder_features": 10,
    "decoder_architecture": "cnn_ts_dec", # not used
    "decoder_features": 0, # not used
    "decoder_layers": 0, # not used
    "max_epochs": 30,
    "training_dir": "_wisdm_clr_cpmodel",
    "cf": 0,
    "same_class": True
}

wisdm_seg_class = {
    "dataset": "WISDM",
    "subjects_for_test": cross_validate["WISDM"],
    "window_size": 30,
    "window_stride": 1,
    "mode": "ts",
    "encoder_architecture": "cnn_gap_ts",
    "encoder_features": 24,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "max_epochs": 30,
    "training_dir": "_wisdm_seg_class",
    "same_class": True
}

experiments = [
    harth_ts, harth_tr, harth_img, harth_seg, # harth_img_g,
    #uci_ts, uci_tr, uci_img, uci_seg,
    #wisdm_ts, wisdm_tr, wisdm_img, wisdm_seg
]#, harth_clr, harth_seg_class, wisdm_clr, wisdm_seg_class]
