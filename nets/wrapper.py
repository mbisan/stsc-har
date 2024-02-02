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
from transforms.dtw import dtw_mode

class WrapperModel(LightningModule):

    def __init__(self, mode, encoder_arch, decoder_arch,
        n_dims, n_classes, n_patterns, l_patterns,
        wdw_len, wdw_str,
        enc_feats, dec_feats, dec_layers, lr, voting, 
        weight_decayL1, weight_decayL2,
        name=None) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        if name is None:
            name = "test"

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())
        self.save_hyperparameters()

        # create encoder
        if mode == "img":
            ref_size, channels = l_patterns, n_patterns
            self.dsrc = "frame"
        elif mode == "ts":
            ref_size, channels = 1, n_dims
            self.dsrc = "series"
        elif mode == "fft":
            ref_size, channels = 1, n_dims*2
            self.dsrc = "transformed"
        elif mode == "dtw":
            ref_size, channels = l_patterns, enc_feats
            self.wdw_len = wdw_len-l_patterns
            self.dsrc = "series"
        elif mode == "dtw_c":
            ref_size, channels = l_patterns, enc_feats*n_dims
            self.wdw_len = wdw_len-l_patterns
            self.dsrc = "series"
        elif mode == "dtwfeats":
            ref_size, channels = 1, enc_feats
            self.wdw_len = 1
            self.dsrc = "series"
        elif mode == "dtwfeats_c":
            ref_size, channels = n_dims, enc_feats
            self.wdw_len = 1
            self.dsrc = "series"
        elif mode in ["mtf", "gasf", "gadf", "cwt_test"]:
            ref_size, channels = wdw_len, n_dims
            self.dsrc = "transformed"

        self.initial_transform = None
        if mode in ["dtw", "dtw_c"]:
            self.initial_transform = dtw_mode[mode](n_patts=enc_feats, d_patts=n_dims, l_patts=l_patterns, l_out=wdw_len-l_patterns, rho=self.voting["rho"]/10)
        elif mode == "dtwfeats" or mode == "dtwfeats_c":
            self.initial_transform = dtw_mode[mode](n_patts=enc_feats, d_patts=n_dims, l_patts=l_patterns, l_out=wdw_len-l_patterns, rho=self.voting["rho"])

        self.encoder = encoder_dict[encoder_arch](channels=channels, ref_size=ref_size, 
            wdw_size=self.wdw_len, n_feature_maps=enc_feats)
        
        # create decoder
        shape: torch.Tensor = self.encoder.get_output_shape()

        inp_feats = torch.prod(torch.tensor(shape[1:]))
        out_feats = n_classes
        self.decoder = decoder_dict[decoder_arch](inp_feats=inp_feats, 
            hid_feats=dec_feats, out_feats=out_feats, hid_layers=dec_layers)

        # create softmax and flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        for phase in ["train", "val", "test"]: 
            self.__setattr__(f"{phase}_cm", tm.ConfusionMatrix(num_classes=out_feats, task="multiclass"))
            if phase != "train":
                self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass", average="macro"))

        self.voting = None
        if voting["n"] > 1:
            self.voting = voting
            self.voting["weights"] = (self.voting["rho"] ** (1/self.wdw_len)) ** torch.arange(self.voting["n"] - 1, -1, -1)

        self.previous_predictions = None
        self.probabilities = []
        self.labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        x=self.logits(x)
        x = self.softmax(x)
        return x
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initial_transform is None:
            x = self.initial_transform(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.logits(batch[self.dsrc])

        # Compute the loss and metrics
        loss = F.cross_entropy(output, batch["label"])

        if stage == "train" or self.voting is None:
            predictions = torch.argmax(output, dim=1)
        if stage != "train" and not self.voting is None:
            pred_prob = torch.softmax(output, dim=1)

            if self.previous_predictions is None:
                pred_ = torch.cat((torch.zeros((self.voting["n"]-1, self.n_classes)), pred_prob), dim=0)
            else:
                pred_ = torch.cat((self.previous_predictions, pred_prob), dim=0)
            
            self.previous_predictions = pred_prob[-(self.voting["n"]-1):,:]

            predictions_weighted = torch.conv2d(pred_[None, None, ...], self.voting["weights"][None, None, :, None])[0, 0]
            predictions = predictions_weighted.argmax(dim=1)

            self.probabilities.append(pred_prob)
            self.labels.append(batch["label"])

        self.__getattr__(f"{stage}_cm").update(predictions, batch["label"])
        if stage != "train" and self.voting is None:
            self.probabilities.append(torch.softmax(output, dim=1))
            self.labels.append(batch["label"])

        # log loss and metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

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

        TP = cm.diag()
        FP = cm.sum(0) - TP
        FN = cm.sum(1) - TP
        TN = torch.empty(cm.shape[0])
        for i in range(cm.shape[0]):
            TN[i] = cm[:i,:i].sum() + cm[:i,i:].sum() + cm[i:,:i].sum() + cm[i:,i:].sum()

        precision = TP/(TP+FP)
        recall = TP/(TP+FN) # this is the same as accuracy per class
        f1 = 2*(precision*recall)/(precision + recall)
        iou = TP/(TP+FP+FN) # iou per class

        self.log(f"{stage}_pr", precision.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_re", recall.nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_f1", f1.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_iou", iou.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)

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