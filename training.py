import os
from time import time

from utils.helper_functions import load_dm, str_time, cm_str
from utils.methods import train_model

from utils.arguments import get_parser, get_model_name

from nets.wrapper import DFWrapper, SegWrapper, ContrastiveWrapper

from pytorch_lightning import seed_everything
import numpy as np

import json

import warnings # shut up warnings
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
            args.lr, args.weight_decayL1, args.weight_decayL2, modelname, window_size=args.window_size, 
            output_regularizer=args.cf, mode=args.mode, monitor="val_aupr")
        modeltype = ContrastiveWrapper
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
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        }, 
        metrics={
            "target": model.monitor, "mode": "max"
        }, modeltype=modeltype)
    
    data = {**data, "args": args.__dict__, "name": modelname}
    data["cm"] = cm_str(data["cm"])
    data = json.dumps(data, indent=2).replace("<>]\"", "]").replace("\"<>", "").replace("\\n", "\n")
    with open(os.path.join(args.training_dir, modeldir, "results.json"), "w") as f:
        f.write(data)
    
    print(data)

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(42)
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
