import os

# def str_time(b):
#     timestr = f"{int(b//3600):02d}:{int((b%3600)//60):02d}"
#     return timestr + f":{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

from data_2.dfdataset import PatternConf
from data_2.datamodule import STSDataModule

def cm_str(cm):
    if cm is None:
        return ""
    str_res = "<>[\n"
    for i in range(len(cm)):
        str_res += "    [" + ",".join([f"{cm[i][j]:>6}" for j in range(len(cm[i]))]) + "],\n"
    str_res = str_res[:-2] + "\n  <>]"
    return str_res

def load_dm(args):
    dm = STSDataModule(
        args.dataset,
        os.path.join(args.dataset_dir, args.dataset),
        args.window_size,
        args.window_stride,
        args.batch_size,
        args.random_seed,
        args.num_workers,
        args.label_mode,
        args.reduce_imbalance,
        args.same_class,
        args.subjects_for_test,
        args.n_val_subjects,
        args.overlap,
        PatternConf(args.pattern_type, args.pattern_size, args.rho, args.cached, args.compute_n),
        args.use_triplets
    )

    message = f"Using {len(dm.train_dataset)} observations for training"
    print(message + f", {len(dm.val_dataset)} for validation and {len(dm.test_dataset)} for test")

    return dm
