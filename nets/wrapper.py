""" Wrapper model for the deep learning models. """

# modules
from nets.baseWrapper import BaseWrapper

# base torch
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch

from nets import encoder_dict, decoder_dict, segmentation_dict
from transforms.dtw import dtw_mode

from nets.losses import SupConLoss, ContrastiveDist, TripletLoss

from sklearn.metrics import roc_auc_score, average_precision_score

class DFWrapper(BaseWrapper):

    def __init__(self, mode, encoder_arch, decoder_arch,
        n_dims, n_classes, n_patterns, l_patterns,
        wdw_len, wdw_str,
        enc_feats, dec_feats, dec_layers, lr, voting, 
        weight_decayL1, weight_decayL2,
        name="test", **kwargs) -> None:

        # save parameters as attributes
        super().__init__(lr, weight_decayL1, weight_decayL2, n_classes, **kwargs), self.__dict__.update(locals())
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
        self.decoder = decoder_dict[decoder_arch](inp_feats=inp_feats, 
            hid_feats=dec_feats, out_feats=n_classes, hid_layers=dec_layers)

        self.voting = None
        if voting["n"] > 1:
            self.voting = voting
            self.voting["weights"] = (self.voting["rho"] ** (1/self.wdw_len)) ** torch.arange(self.voting["n"] - 1, -1, -1)

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
        loss = F.cross_entropy(output, batch["label"], ignore_index=100)

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
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=stage=="train", prog_bar=True, logger=True)

        # return loss
        if stage == "train":
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)

class SegWrapper(BaseWrapper):

    def __init__(self, in_channels, latent_features, n_classes, 
            pooling, kernel_size, complexity_factor, lr, weight_decayL1, weight_decayL2, arch, name=None, overlap=1, **kwargs) -> None:

        # save parameters as attributes
        super().__init__(lr, weight_decayL1, weight_decayL2, n_classes, **kwargs), self.__dict__.update(locals())
        self.save_hyperparameters()

        if "unet" in arch:
            self.segmentation = segmentation_dict[arch](in_channels, n_classes, latent_features)
        elif "utime" in arch:
            self.segmentation = segmentation_dict[arch](n_classes=n_classes, in_dims=in_channels, depth = len(pooling), 
                dilation = 1, kernel_size = kernel_size, padding = "same", init_filters = latent_features, 
                complexity_factor = complexity_factor, pools = pooling, segment_size = 1, change_size = kernel_size)
        elif "dlv3" in arch:
            self.segmentation = segmentation_dict[arch](in_channels, latent_features, n_classes, pooling)

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
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)

class ContrastiveWrapper(BaseWrapper):

    def __init__(self, encoder_arch, in_channels, latent_features, lr, weight_decayL1, weight_decayL2, 
        name=None, window_size=8, output_regularizer=0.01, mode="clr", **kwargs) -> None:

        # save parameters as attributes
        super().__init__(lr, weight_decayL1, weight_decayL2, 2, **kwargs), self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = encoder_dict[encoder_arch](
            channels=in_channels, ref_size=0, 
            wdw_size=window_size, n_feature_maps=latent_features
        )

        self.flatten = nn.Flatten()
        output_shape = self.encoder.get_output_shape()
        features = torch.prod(torch.tensor(output_shape[1:]))

        self.mlp1 = decoder_dict["mlp"](inp_feats = features, hid_feats = latent_features*2, out_feats = latent_features*2, hid_layers = 1)
        self.mlp2 = decoder_dict["mlp"](inp_feats = latent_features*2, hid_feats = latent_features*2, out_feats = latent_features, hid_layers = 1)

        # self.project = decoder_dict["mlp"](inp_feats = latent_features*2, hid_feats = latent_features, out_feats = latent_features, hid_layers = 1)

        if mode == "clr3":
            self.contrastive_loss = TripletLoss(epsilon=1e-6, m=2.0)
        elif mode == "clr":
            self.contrastive_loss = ContrastiveDist(epsilon=1e-6, m=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x) 
        x = self.flatten(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        # x must be a (n, d) dimensional matrix
        # x = F.normalize(x, p=2, dim=-1)
        return x # projection module is only used while training

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        if stage == "train":
            output = self.forward(batch["triplet"].view(-1, batch["triplet"].shape[-2], batch["triplet"].shape[-1]))
            output = output.view(-1, 3, output.shape[-1])
        else:
            output = self.forward(batch["series"])

        if stage != "train":
            self.probabilities.append(output)
            self.labels.append(batch["label"])

        # Compute the loss and metrics
        loss = self.contrastive_loss(output, labels=batch["label"])

        # log loss and metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=stage=="train", prog_bar=True, logger=True)

        loss = loss + self.output_regularizer*output.square().sum(-1).sqrt().mean()

        # return loss
        if stage == "train":
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)

    def log_metrics(self, stage):
        if stage=="train":
            return

        representations = torch.concatenate(self.probabilities, dim=0) 
        all_labels = torch.concatenate(self.labels, dim=0)

        diff = (representations[:-self.window_size, :] - representations[self.window_size:, :]).square().sum(-1) + 1e-8
        dissimilarities = diff.sqrt()
        labels = (all_labels[:-self.window_size] != all_labels[self.window_size:]).long()

        dissimilarities = dissimilarities.cpu().numpy()
        labels = labels.cpu().numpy()

        try:
            auroc = roc_auc_score(labels, dissimilarities)
            aupr = average_precision_score(labels, dissimilarities)
        except:
            auroc = 0
            aupr = 0

        self.dissimilarities = dissimilarities
        self.labels_ = labels

        if stage == "test":
            self.rpr = representations

        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_aupr", aupr, on_epoch=True, on_step=False, prog_bar=True, logger=True)
