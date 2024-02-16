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

from nets import encoder_dict, decoder_dict
from nets.metrics import metrics_from_cm, print_cm
from nets.losses import SupConLoss

class ContrastiveModel(LightningModule):

    # following http://arxiv.org/abs/2004.11362 : Supervised Contrastive Learning

    def __init__(self, encoder_arch, in_channels, latent_features, lr, weight_decayL1, weight_decayL2, name=None) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        if name is None:
            name = "test"

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = encoder_dict[encoder_arch](
            channels=in_channels, ref_size=0, 
            wdw_size=32, n_feature_maps=latent_features
        )

        self.flatten = nn.Flatten()
        output_shape = self.encoder.get_output_shape()
        features = torch.prod(torch.tensor(output_shape[1:]))

        self.encoder_2 = decoder_dict["mlp"](inp_feats = features, hid_feats = latent_features*2, out_feats = latent_features*2, hid_layers = 1)

        self.project = decoder_dict["mlp"](inp_feats = latent_features*2, hid_feats = latent_features, out_feats = latent_features, hid_layers = 1)

        self.contrastive_loss = SupConLoss()

        for phase in ["train", "val", "test"]:
            # TODO pensar como evaluar, lo mejor será usar un AUPR or AUROC de dos clases
            # self.__setattr__(f"{phase}_cm", tm.ConfusionMatrix(num_classes=n_classes, task="multiclass", ignore_index=100))
            # if phase != "train":
            #     self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_classes, task="multiclass", average="macro", ignore_index=100))
            pass

        self.previous_predictions = None
        self.probabilities = []
        self.labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x) 
        x = self.flatten(x)
        x = self.encoder_2(x)
        # x must be a (n, d) dimensional matrix
        x = F.normalize(x, p=2, dim=-1)
        return x # projection module is only used while training

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.forward(batch["series"])
        output_p = self.project(output)
        output_p = F.normalize(output_p, p=2, dim=-1)

        # Compute the loss and metrics
        loss = self.contrastive_loss(output_p.unsqueeze(1), labels=batch["label"])

        if stage != "train":
            self.probabilities.append(output_p)
            self.labels.append(batch["series"])

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
        # auc_per_class = tm.functional.auroc(
        #     torch.concatenate(self.probabilities, dim=0), 
        #     torch.concatenate(self.labels, dim=0), 
        #     task="multiclass",
        #     num_classes=self.n_classes)
        # self.probabilities = []
        # self.labels = []

        # self.log(f"{stage}_auroc", auc_per_class.nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        self.log_metrics("val")

    def on_test_epoch_end(self):
        self.log_metrics("test")

    def predict_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        return self(batch[self.dsrc])

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "min"
        monitor = "val_loss"
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