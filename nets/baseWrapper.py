""" Wrapper model for the deep learning models. """

# pylint: disable=invalid-name

# modules
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

from pytorch_lightning import LightningModule
import torchmetrics as tm

from nets.metrics import metrics_from_cm

# pylint: disable=too-many-instance-attributes

class BaseWrapper(LightningModule):

    '''
        Must define init, logits, forward, innerstep and log_metrics

        For multiclass classification, computes and logs precision, recall and computes cm
        for other metrics re-define log_metrics
    '''

    # pylint: disable=unused-argument unnecessary-dunder-call arguments-differ

    def __init__(self, lr, weight_decayL1, weight_decayL2, num_classes, **kwargs) -> None:

        # save parameters as attributes
        super().__init__()

        self.lr = lr
        self.dsrc = "ts"
        self.monitor = kwargs.get("monitor", "val_re")
        self.optimizer_mode = kwargs.get("optimizer_mode", "max")
        self.previous_predictions = None
        self.probabilities = []
        self.labels = []
        self.weight_decayL1 = weight_decayL1
        self.weight_decayL2 = weight_decayL2
        self.cm_last = torch.Tensor([])

        # create softmax and flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        for phase in ["train", "val", "test"]:
            self.__setattr__(f"{phase}_cm",
                tm.ConfusionMatrix(num_classes=num_classes, task="multiclass", ignore_index=100))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):
        pass

    def regularizer_loss(self):
        l1_loss = torch.tensor(0., requires_grad=True)
        l2_loss = torch.tensor(0., requires_grad=True)
        if self.weight_decayL1 > 0:
            l1_loss = self.weight_decayL1 * sum(
                p.abs().sum() for
                name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))

        if self.weight_decayL2 > 0:
            l2_loss = self.weight_decayL2 * sum(
                p.square().sum() for
                name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))

        return l1_loss.to(torch.float32) + l2_loss.to(torch.float32)

    def training_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")

    def validation_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    def on_train_epoch_end(self):
        self.log_metrics("train")
        self.probabilities = []
        self.labels = []

    def on_validation_epoch_end(self):
        self.log_metrics("val")
        self.probabilities = []
        self.labels = []

    def on_test_epoch_end(self):
        self.log_metrics("test")
        self.probabilities = []
        self.labels = []

    def configure_optimizers(self):
        """ Configure the optimizers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode=self.optimizer_mode, factor=np.sqrt(0.1), patience=2, min_lr=1e-6),
                "interval": "epoch",
                "monitor": self.monitor,
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def log_metrics(self, stage):
        cm = self.__getattr__(f"{stage}_cm").compute()
        self.__getattr__(f"{stage}_cm").reset()

        metrics = metrics_from_cm(cm)

        self.log(
            f"{stage}_pr", metrics["precision"].nanmean(), on_epoch=True,
            on_step=False, prog_bar=False, logger=True)
        self.log(
            f"{stage}_re", metrics["recall"].nanmean(), on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            f"{stage}_f1", metrics["f1"].nanmean(), on_epoch=True,
            on_step=False, prog_bar=False, logger=True)
        self.log(
            f"{stage}_iou", metrics["iou"].nanmean(), on_epoch=True,
            on_step=False, prog_bar=False, logger=True)
        self.log(
            f"{stage}_acc", metrics["accuracy"].nanmean(), on_epoch=True,
            on_step=False, prog_bar=False, logger=True)

        if stage == "test":
            self.cm_last = cm

        if stage != "train":
            auc_per_class = tm.functional.auroc(
                torch.concatenate(self.probabilities, dim=0),
                torch.concatenate(self.labels, dim=0),
                task="multiclass",
                num_classes=self.n_classes,
                ignore_index=100)
            self.probabilities = []
            self.labels = []

            self.log(
                f"{stage}_auroc", auc_per_class.nanmean(), on_epoch=True,
                on_step=False, prog_bar=True, logger=True)
