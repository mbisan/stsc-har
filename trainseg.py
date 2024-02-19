import os
from time import time

from utils.helper_functions import load_dm, get_parser, str_time
from utils.methods import train_model

from nets.wrapper import SegWrapper

from utils.arguments import get_model_name_seg

from pytorch_lightning import seed_everything

import warnings # shut up warnings
warnings.simplefilter("ignore", category=UserWarning)

def main(args):
    dm = load_dm(args)

    modelname = get_model_name_seg(args)
    print("\n" + modelname)
    modeldir = modelname.replace("|", "_").replace(",", "_")

    model = SegWrapper(
        dm.n_dims, args.encoder_features, dm.n_classes, args.pooling, args.pattern_size, 
        args.cf, args.lr, args.weight_decayL1, args.weight_decayL2, args.encoder_architecture, modelname, args.overlap)
    
    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        }, 
        metrics={
            "target": "val_re", "mode": "max"
        }, modeltype=SegWrapper)
    
    with open(os.path.join(args.training_dir, modeldir, "results.dict"), "w") as f:
        f.write(str({**data, **args.__dict__, "name": modelname}))
    
    print(data)

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(42)
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
