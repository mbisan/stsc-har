import os
from time import time

from utils.helper_functions import load_dm, str_time, cm_str
from utils.methods import train_model

from nets.wrapper import ContrastiveWrapper

from pytorch_lightning import seed_everything
import numpy as np
import json

from utils.arguments import get_parser, get_model_name

import warnings # shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def main(args):
    dm = load_dm(args)

    modelname = get_model_name(args)
    modeldir = modelname.replace("|", "_").replace(",", "_")

    print("\n" + modelname)

    model = ContrastiveWrapper(
        args.encoder_architecture, dm.n_dims, args.encoder_features, 
        args.lr, args.weight_decayL1, args.weight_decayL2, modelname, window_size=args.window_size, 
        output_regularizer=args.cf, mode=args.mode, monitor="val_aupr")
    
    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        },
        metrics={
            "target": model.monitor, "mode": "max"
        },
        modeltype=ContrastiveWrapper)
    
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
