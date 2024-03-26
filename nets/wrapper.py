""" Wrapper model for the deep learning models. """

# modules
import torch
from torch.nn import functional as F
import torch.nn as nn

from sklearn.metrics import average_precision_score, roc_curve, auc

from transforms.gaf_mtf import GAFLayer, MTFLayer

from nets import encoder_dict, decoder_dict, segmentation_dict
from nets.baseWrapper import BaseWrapper
from nets.losses import SupConLoss, TripletLoss

from utils.arguments import Arguments

# pylint: disable=too-many-ancestors too-many-arguments too-many-locals

def get_encoder(
        n_dims, n_classes, n_patterns, l_patterns, args: Arguments, **kwargs
    ):
    '''
        Returns tuple with:
            data source, encoder nn.Module, initial transform nn.module
    '''
    # pylint: disable=unused-argument too-many-return-statements
    if args.mode=="df":
        return (
            "frame", 
            encoder_dict[args.encoder_architecture](
                channels=n_patterns, ref_size=l_patterns,
                wdw_size=args.window_size, n_feature_maps=args.encoder_features), None)
    if args.mode=="ts":
        return (
            "series", 
            encoder_dict[args.encoder_architecture](
                channels=n_dims, ref_size=1,
                wdw_size=args.window_size, n_feature_maps=args.encoder_features), None)
    if args.mode == "mtf":
        return (
            "series",
            encoder_dict[args.encoder_architecture](
                channels=n_dims, ref_size=args.window_size,
                wdw_size=args.window_size, n_feature_maps=args.encoder_features),
            MTFLayer(args.mtf_bins, (-1, 1)) )
    if args.mode in ["gasf", "gadf"]:
        return (
            "series",
            encoder_dict[args.encoder_architecture](
                channels=n_dims, ref_size=args.window_size,
                wdw_size=args.window_size, n_feature_maps=args.encoder_features),
            GAFLayer(args.mode, (-1, 1)) )
    if args.mode == "rnn":
        return (
            "series",
            encoder_dict[args.encoder_architecture](
                channels=n_dims, latent_size=args.encoder_features, n_layers=args.encoder_layers),
            None )
    # TODO
    if args.mode=="fft":
        return (
            "series", 
            encoder_dict[args.encoder_architecture](
                channels=n_dims, ref_size=1,
                wdw_size=args.window_size, n_feature_maps=args.encoder_features), None)
    if args.mode=="dtw":
        return (
            "series",
            None, None)
    if args.mode=="dtw_c":
        return (
            "series",
            None, None)
    if args.mode == "dtwfeats":
        return (
            "series",
            None, None)
    if args.mode == "dtwfeats_c":
        return (
            "series",
            None, None)
    if args.mode == "seg":
        if args.encoder_architecture == "utime":
            return (
                "series",
                segmentation_dict["utime"](n_classes=n_classes, in_dims=n_dims,
                    depth = len(args.pooling), dilation = 1, kernel_size = args.pattern_size,
                    padding = "same", init_filters = args.encoder_features,
                    complexity_factor = args.cf, pools = args.pooling,
                    segment_size = 1, change_size = args.pattern_size), None)
        return (
            "series",
            segmentation_dict[args.encoder_architecture](
                n_dims, n_classes, args.encoder_features, args.pooling
            ), None)
    return None

def get_decoder(n_dims, n_classes, inp_feats, args: Arguments, **kwargs):
    '''
        Returns tuple with:
            label source, decoder nn.Module, label transform
    '''
    # pylint: disable=unused-argument too-many-return-statements
    lbsrc = "label"
    if args.mode in ["seg"]:
        lbsrc = "scs"

    return (
        lbsrc, decoder_dict[args.decoder_architecture](
            inp_feats = torch.prod(torch.tensor(inp_feats)),
            hid_feats = args.decoder_features, out_feats = n_classes,
            hid_layers = args.decoder_layers))


class ClassifierWrapper(BaseWrapper):

    # pylint: disable=unused-argument unnecessary-dunder-call arguments-differ

    def __init__(self,
        n_dims, n_classes, n_patterns, l_patterns, args: Arguments, name="test", **kwargs) -> None:

        # save parameters as attributes
        super().__init__(
            args.lr, args.weight_decayL1, args.weight_decayL2,
            n_classes, **kwargs)

        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.dsrc, self.encoder, self.initial_transform = get_encoder(
            n_dims, n_classes, n_patterns, l_patterns, args
        )

        # get latent size of encoder
        if self.dsrc == "frame":
            x_test = torch.randn((1, n_patterns, l_patterns, args.window_size))
        else: # elif "series"
            x_test = torch.randn((1, n_dims, args.window_size))
        if not self.initial_transform is None:
            x_test = self.initial_transform(x_test)
        x_test = self.encoder(x_test)
        print("Latent shape:", tuple(x_test.shape[1:]))

        self.lbsrc, self.decoder = get_decoder(
            n_dims, n_classes, x_test.shape[1:], args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        x = self.logits(x)
        x = self.softmax(x)
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initial_transform is None:
            x = self.initial_transform(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.logits(batch[self.dsrc])

        # Compute the loss and metrics
        loss = F.cross_entropy(output, batch[self.lbsrc], ignore_index=100)

        predictions = torch.argmax(output, dim=1)
        self.__getattr__(f"{stage}_cm").update(predictions, batch[self.lbsrc])

        if stage != "train":
            self.probabilities.append(torch.softmax(output, dim=1))
            self.labels.append(batch[self.lbsrc])

        # log loss and metrics
        self.log(
            f"{stage}_loss", loss, on_epoch=True,
            on_step=stage=="train", prog_bar=True, logger=True)

        # return loss
        if stage == "train":
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)


class ContrastiveWrapper(BaseWrapper):

    # pylint: disable=unused-argument unnecessary-dunder-call arguments-differ

    def __init__(self, encoder_arch, in_channels, latent_features, lr,
            weight_decayL1, weight_decayL2,
            name=None, window_size=8, output_regularizer=0.01,
            mode="clr", overlap=0, **kwargs) -> None:

        # save parameters as attributes
        super().__init__(lr, weight_decayL1, weight_decayL2, 2, **kwargs)

        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = encoder_dict[encoder_arch](
            channels=in_channels, ref_size=0,
            wdw_size=window_size, n_feature_maps=latent_features
        )

        self.flatten = nn.Flatten()
        output_shape = self.encoder.get_output_shape()
        features = torch.prod(torch.tensor(output_shape[1:]))

        self.mlp1 = decoder_dict["mlp"](
            inp_feats = features, hid_feats = latent_features*2,
            out_feats = latent_features*2, hid_layers = 1)
        self.mlp2 = decoder_dict["mlp"](
            inp_feats = latent_features*2, hid_feats = latent_features*2,
            out_feats = latent_features, hid_layers = 1)

        # self.project = decoder_dict["mlp"](
        #     inp_feats = latent_features*2, hid_feats = latent_features,
        #     out_feats = latent_features, hid_layers = 1)

        if mode == "clr3":
            self.contrastive_loss = TripletLoss(epsilon=1e-6, m=5.0)
        elif mode == "clr":
            self.contrastive_loss = SupConLoss()

        self.dissimilarities = None
        self.labels_ = None
        self.rpr = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        # x must be a (n, d) dimensional matrix
        if self.mode == "clr":
            x = F.normalize(x, p=2, dim=-1)
        return x # projection module is only used while training

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        if stage == "train":
            output = self.forward(
                batch["triplet"].view(-1, batch["triplet"].shape[-2], batch["triplet"].shape[-1]))
            output = output.view(-1, 3, output.shape[-1])
        else:
            output = self.forward(batch["series"])

        if stage != "train":
            self.probabilities.append(output)
            self.labels.append(batch["change_point"])

        # Compute the loss and metrics
        if stage == "train":
            loss = self.contrastive_loss(output)
        else:
            loss = 0

        # log loss and metrics
        self.log(
            f"{stage}_loss", loss, on_epoch=True,
            on_step=stage=="train", prog_bar=True, logger=True)

        loss = loss + self.output_regularizer*output.square().sum(-1).sqrt().mean()

        # return loss
        if stage == "train":
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)

    def log_metrics(self, stage):
        if stage=="train":
            return

        representations = torch.concatenate(self.probabilities, dim=0)
        all_labels = torch.concatenate(self.labels, dim=0).long()

        displacement = self.window_size-self.overlap
        if self.mode == "clr":
            dissimilarities = - (
                representations[:(-displacement), :] * representations[displacement:, :]).sum(-1)
        else:
            diff = (
                representations[:(-displacement), :] - representations[displacement:, :])
            dissimilarities = (diff.square().sum(-1) + 1e-8).sqrt()

        dissimilarities = dissimilarities.cpu().numpy()
        labels = all_labels[(self.window_size-self.overlap):].cpu().numpy()

        try:
            fpr, tpr, thresholds = roc_curve(labels, dissimilarities)
            auroc = auc(fpr, tpr)

            for i, tpr_ in enumerate(tpr):
                if tpr_ > 0.95:
                    self.log(
                        f"{stage}_fpr95", fpr[i], on_epoch=True,
                        on_step=False, prog_bar=False, logger=True)
                    self.log(
                        f"{stage}_th", thresholds[i], on_epoch=True,
                        on_step=False, prog_bar=False, logger=True)
                    break

            aupr = average_precision_score(labels, dissimilarities)
        except ValueError:
            print("Error occurred")
            auroc = 0
            aupr = 0

        self.dissimilarities = dissimilarities
        self.labels_ = labels

        if stage == "test":
            self.rpr = representations

        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_ap", aupr, on_epoch=True, on_step=False, prog_bar=True, logger=True)



class AutoencoderWrapper(BaseWrapper):

    # pylint: disable=unused-argument unnecessary-dunder-call arguments-differ

    def __init__(self, encoder_arch, in_channels, latent_features, decoder_arch,
            lr, weight_decayL1, weight_decayL2, latent_regularizer=0.01,
            name=None, window_size=8, **kwargs) -> None:

        # save parameters as attributes
        super().__init__(lr, weight_decayL1, weight_decayL2, 2, **kwargs)

        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = encoder_dict[encoder_arch](
            channels=in_channels, ref_size=0,
            wdw_size=window_size, n_feature_maps=latent_features
        )

        output_shape = self.encoder.get_output_shape()

        self.decoder = decoder_dict[decoder_arch](
            inp_feats=output_shape[1], hid_feats=latent_features, out_feats=in_channels
        )

        self.dissimilarities = None
        self.labels_ = None
        self.rpr = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        y = self.decoder(x)
        return {"reconstruction": y, "latent": x}

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.forward(batch["series"])
        diff = output["reconstruction"] - batch["series"]
        reconstruction_error = diff.square().sum(dim=(-2, -1))

        if stage != "train":
            self.rpr.append(output["latent"])
            self.probabilities.append(reconstruction_error) # save the reconstruction difference
            # from batch["scs"] we can extract if there is a change point in the window
            # by simply checking how many unique labels are there in the window
            self.labels.append(batch["change_point"])

        # Compute the loss (MSE)
        loss = reconstruction_error.mean()

        # log loss and metrics
        self.log(
            f"{stage}_loss", loss, on_epoch=True,
            on_step=stage=="train", prog_bar=True, logger=True)

        loss = loss + self.latent_regularizer * output["latent"].sum(dim=(-2, -1)).mean()

        # return loss
        if stage == "train":
            return loss.to(torch.float32) + self.regularizer_loss()

        return loss.to(torch.float32)

    def log_metrics(self, stage):
        if stage=="train":
            return

        # dim (n, channels, window_size)
        reconstruction_errors = torch.concatenate(self.probabilities, dim=0)
        labels = torch.concatenate(self.labels, dim=0).long()

        reconstruction_errors = reconstruction_errors.cpu().numpy()
        labels = labels.cpu().numpy()

        try:
            fpr, tpr, thresholds = roc_curve(labels, reconstruction_errors)
            auroc = auc(fpr, tpr)

            for i, tpr_ in enumerate(tpr):
                if tpr_ > 0.95:
                    self.log(
                        f"{stage}_fpr95", fpr[i], on_epoch=True,
                        on_step=False, prog_bar=False, logger=True)
                    self.log(
                        f"{stage}_th", thresholds[i], on_epoch=True,
                        on_step=False, prog_bar=False, logger=True)
                    break

            aupr = average_precision_score(labels, reconstruction_errors)
        except ValueError:
            auroc = 0
            aupr = 0

        self.dissimilarities = reconstruction_errors
        self.labels_ = labels

        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_ap", aupr, on_epoch=True, on_step=False, prog_bar=True, logger=True)
