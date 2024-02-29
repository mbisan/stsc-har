from collections import namedtuple

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything, LightningModule

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

PLKWargs = namedtuple("PL_KWargs", ["default_root_dir", "accelerator", "seed"])
MetricsSetting = namedtuple("MetricsSetting", ["target", "mode"])

def train_model(
        dm,
        model,
        max_epochs: int,
        pl_kwargs: PLKWargs = PLKWargs("training", "auto", 42),
        metrics: MetricsSetting = MetricsSetting("val_re", "max"),
        modeltype = LightningModule) -> tuple[LightningModule, dict]:

    # pylint: disable=too-many-arguments

    # reset the random seed
    seed_everything(pl_kwargs.seed, workers=True)

    # set up the trainer
    ckpt = ModelCheckpoint(
        monitor=metrics.target,
        mode=metrics.mode,
        save_top_k=3,
        filename='{epoch}-{step}-{val_re:.4f}'
    )

    tr = Trainer(
        default_root_dir=pl_kwargs.default_root_dir,
        accelerator=pl_kwargs.accelerator,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=25)],
        max_epochs=max_epochs,
        logger=TensorBoardLogger(
            save_dir=pl_kwargs.default_root_dir,
            name=model.name.replace("|", "_").replace(",", "_")
        )
    )

    # train the model
    tr.fit(model=model, datamodule=dm)

    # load the best weights
    print(ckpt.best_model_path)
    model = modeltype.load_from_checkpoint(ckpt.best_model_path)

    # run the validation with the final weights
    data = tr.test(model, datamodule=dm, verbose=False)

    return_data = {
        **data[0],
        "cm": model.cm_last.tolist() if hasattr(model, "cm_last") else None,
        "path": ckpt.best_model_path
    }

    return model, return_data
