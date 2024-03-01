import os
from time import time

import json
import subprocess

import warnings

from pytorch_lightning import seed_everything
import numpy as np

from utils.helper_functions import load_dm, str_time, cm_str
from utils.methods import train_model, PLKWargs, MetricsSetting

from utils.arguments import get_parser, get_model_name

from nets.wrapper import DFWrapper, SegWrapper, ContrastiveWrapper, AutoencoderWrapper

# shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def main(args):
    dm = load_dm(args)

    modelname = get_model_name(args)
    modeldir = modelname.replace("|", "_").replace(",", "_")

    print("\n" + modelname)

    if args.mode == "seg":
        model = SegWrapper(
            dm.n_dims, args.encoder_features, dm.n_classes, args.pooling, args.pattern_size,
            args.cf, args.lr, args.weight_decayL1, args.weight_decayL2, args.encoder_architecture,
            modelname, args.overlap if args.overlap>=0 else args.window_size - 1, monitor="val_re")
        modeltype = SegWrapper
    elif "clr" in args.mode:
        model = ContrastiveWrapper(
            args.encoder_architecture, dm.n_dims, args.encoder_features,
            args.lr, args.weight_decayL1, args.weight_decayL2, modelname,
            window_size=args.window_size,
            output_regularizer=args.cf, mode=args.mode, monitor="val_ap")
        modeltype = ContrastiveWrapper
    elif args.mode == "ae":
        model = AutoencoderWrapper(
            args.encoder_architecture, dm.n_dims, args.encoder_features, args.decoder_architecture,
            args.lr, args.weight_decayL1, args.weight_decayL2, args.cf, modelname, args.window_size,
            monitor="val_ap"
        )
        modeltype = AutoencoderWrapper
    else:
        model = DFWrapper(
            args.mode,
            args.encoder_architecture, args.decoder_architecture,
            dm.n_dims, dm.n_classes, dm.n_patterns, dm.l_patterns, dm.wdw_len, dm.wdw_str,
            args.encoder_features, args.decoder_features, args.decoder_layers,
            args.lr, {"n": args.voting, "rho": args.rho},
            args.weight_decayL1, args.weight_decayL2, modelname, monitor="val_re")
        modeltype = DFWrapper

    # save computed patterns for later use
    if not os.path.exists(os.path.join(args.training_dir)):
        os.mkdir(os.path.join(args.training_dir))
    if not os.path.exists(os.path.join(args.training_dir, modeldir)):
        os.mkdir(os.path.join(args.training_dir, modeldir))
    if hasattr(dm, "dfds"):
        with open(os.path.join(args.training_dir, modeldir, "pattern.npz"), "wb") as f:
            np.save(f, dm.dfds.patterns)

    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs,
        pl_kwargs=PLKWargs(args.training_dir, "auto", 42),
        metrics=MetricsSetting(model.monitor, "max"), modeltype=modeltype)

    data = {
        **data,
        "args": args.__dict__, 
        "name": modelname, 
        "commit": subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    }
    data["cm"] = cm_str(data["cm"]) if len(data["cm"]) > 0 else None
    data = json.dumps(data, indent=2).replace("<>]\"", "]").replace("\"<>", "").replace("\\n", "\n")
    out_file = os.path.join(args.training_dir, modeldir, "results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(data)

    print(data)

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    _args = parser.parse_args()

    seed_everything(42)

    print(_args)
    main(_args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
