from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from nets.wrapper import WrapperModel

def create_model_from_DM(dm, mode: str, enc_arch: str, dec_arch: str, name: str = None,
        enc_feats: int = 32, dec_feats: int = 32, dec_layers: int = None, lr: float = 1e-3, voting: dict = {"n": 1, "w": 1}, 
        weight_decayL1: float = 0, weight_decayL2: float = 0
        ) -> WrapperModel:
        

    return WrapperModel(
        mode=mode, encoder_arch=enc_arch, decoder_arch=dec_arch,
        n_dims=dm.n_dims, n_classes=dm.n_classes, n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
        wdw_len=dm.wdw_len, wdw_str=dm.wdw_str,
        enc_feats=enc_feats, dec_feats=dec_feats, dec_layers=dec_layers, lr=lr, voting=voting, 
        weight_decayL1=weight_decayL1, weight_decayL2=weight_decayL2, name=name
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def train_model(
        dm, 
        model: WrapperModel,
        max_epochs: int,
        pl_kwargs: dict = {"default_root_dir": "training", "accelerator": "auto", "seed": 42}) -> tuple[WrapperModel, dict]:
    
    # reset the random seed
    seed_everything(pl_kwargs["seed"], workers=True)

    # choose metrics
    metrics = {"all": ["re", "f1", "auroc"], "target": "val_re", "mode": "max"}

    # set up the trainer
    ckpt = ModelCheckpoint(
        monitor=metrics['target'], 
        mode=metrics["mode"],
        save_top_k=-1,
        filename='{epoch}-{step}-{val_re:.2f}'
    )

    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"], 
    accelerator=pl_kwargs["accelerator"], callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")], max_epochs=max_epochs,
    logger=TensorBoardLogger(save_dir=pl_kwargs["default_root_dir"], name=model.name.replace("|", "_").replace(",", "_")))

    # train the model
    tr.fit(model=model, datamodule=dm)

    # load the best weights
    model = WrapperModel.load_from_checkpoint(ckpt.best_model_path)

    # run the validation with the final weights
    data = tr.test(model, datamodule=dm)

    return model, data[0]
