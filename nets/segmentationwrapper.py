""" Wrapper model for the deep learning models. """

# modules
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch

from nets.segmentation import segmentation_dict
from nets.metrics import metrics_from_cm, print_cm

class SegmentationModel(LightningModule):

    def __init__(self, in_channels, latent_features, n_classes, pooling, lr, weight_decayL1, weight_decayL2, name=None, overlap=1) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        if name is None:
            name = "test"

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())
        self.save_hyperparameters()

        if "unet" in name:
            self.segmentation = segmentation_dict["unet"](in_channels, n_classes, latent_features) # get_model(in_channels, latent_features, n_classes, aspp_dilate)
        elif "utime" in name:
            self.segmentation = segmentation_dict["utime"](n_classes=n_classes, in_dims=in_channels, depth = len(pooling), 
                dilation = 1, kernel_size = 3, padding = "same", init_filters = latent_features, complexity_factor = 1.5, pools = pooling, segment_size = 1, change_size = 3)
        elif "dlv3" in name:
            self.segmentation = segmentation_dict["dlv3"](in_channels, latent_features, n_classes, pooling)
        self.softmax = nn.Softmax()

        for phase in ["train", "val", "test"]: 
            self.__setattr__(f"{phase}_cm", tm.ConfusionMatrix(num_classes=n_classes, task="multiclass", ignore_index=100))
            if phase != "train":
                self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_classes, task="multiclass", average="macro", ignore_index=100))

        self.previous_predictions = None
        self.probabilities = []
        self.labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.logits(x)
        return self.softmax(x)
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmentation(x)

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.logits(batch["series"])

        skip = output.shape[-1] - self.overlap

        # Compute the loss and metrics
        loss = F.cross_entropy(output, batch["scs"], ignore_index=100)

        predictions = torch.argmax(output, dim=1)[:, -skip:]

        self.__getattr__(f"{stage}_cm").update(predictions, batch["scs"][:, -skip:])
        if stage != "train":
            self.probabilities.append(torch.softmax(output, dim=1)[:, :, -skip:])
            self.labels.append(batch["scs"][:, -skip:])

        # log loss and metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=stage=="train", prog_bar=True, logger=True)

        # return loss
        if stage == "train":
            l1_loss = torch.tensor(0., requires_grad=True)
            l2_loss = torch.tensor(0., requires_grad=True)
            if self.weight_decayL1 > 0:
                l1_loss = self.weight_decayL1 * sum(p.abs().sum() for name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))
                # self.log(f"{stage}_L1", l1_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

            if self.weight_decayL2 > 0:
                l2_loss = self.weight_decayL2 * sum(p.square().sum() for name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))
                # self.log(f"{stage}_L2", l2_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

            return loss.to(torch.float32) + l1_loss.to(torch.float32) + l2_loss.to(torch.float32)

        return loss.to(torch.float32)

    def training_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")
    
    def log_metrics(self, stage):
        cm = self.__getattr__(f"{stage}_cm").compute()
        self.__getattr__(f"{stage}_cm").reset()

        metrics = metrics_from_cm(cm)

        self.log(f"{stage}_pr", metrics["precision"].nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_re", metrics["recall"].nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_f1", metrics["f1"].nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_iou", metrics["iou"].nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)

        if stage == "test":
            print_cm(cm, self.n_classes)

        if stage != "train":
            auc_per_class = tm.functional.auroc(
                torch.concatenate(self.probabilities, dim=0), 
                torch.concatenate(self.labels, dim=0), 
                task="multiclass",
                num_classes=self.n_classes)
            self.probabilities = []
            self.labels = []

            self.log(f"{stage}_auroc", auc_per_class.nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        self.log_metrics("train")

    def on_validation_epoch_end(self):
        self.log_metrics("val")

    def on_test_epoch_end(self):
        self.log_metrics("test")

    def predict_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        return self(batch[self.dsrc])

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "max"
        monitor = "val_re"
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode=mode, factor=np.sqrt(0.1), patience=2, min_lr=1e-6),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }