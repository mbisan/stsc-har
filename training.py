import os
from time import time
from datetime import timedelta

import json
import subprocess

import warnings

from pytorch_lightning import seed_everything

from utils.methods import train_model, PLKWargs, MetricsSetting, load_dm, cm_str
from utils.arguments import get_parser, get_model_name, Arguments

from nets.wrapper import (
    ClassifierWrapper, ContrastiveWrapper, AutoencoderWrapper
)

# shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def main(args: Arguments):
    dm = load_dm(args)

    modelname = get_model_name(args)
    modeldir = modelname.replace("|", "_").replace(",", "_")

    print("\n" + modelname)

    if "clr" in args.mode:
        model = ContrastiveWrapper(
            args.encoder_architecture, dm.n_dims, args.encoder_features,
            args.lr, args.weight_decayL1, args.weight_decayL2, modelname,
            window_size=args.window_size,
            output_regularizer=args.cf, mode=args.mode, overlap=args.voting, monitor="val_ap")
        modeltype = ContrastiveWrapper
    elif args.mode == "ae":
        model = AutoencoderWrapper(
            args.encoder_architecture, dm.n_dims, args.encoder_features, args.decoder_architecture,
            args.lr, args.weight_decayL1, args.weight_decayL2, args.cf, modelname, args.window_size,
            monitor="val_loss", optimizer_mode="min"
        )
        modeltype = AutoencoderWrapper
    else:
        model = ClassifierWrapper(
            dm.n_dims, dm.n_classes, dm.n_patterns, dm.l_patterns,
            args, modelname, monitor="val_re")
        modeltype = ClassifierWrapper

    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs,
        pl_kwargs=PLKWargs(args.training_dir, "auto", args.random_seed),
        metrics=MetricsSetting(model.monitor, model.optimizer_mode), modeltype=modeltype)

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
    _args = Arguments(**vars(parser.parse_args()))

    seed_everything(42)

    print(_args)
    main(_args)
    print(f"Elapsed time: {timedelta(seconds=time()-start_time)}")
