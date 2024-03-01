# pylint: disable=invalid-name

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

class UNET(nn.Module):

    def __init__(self, in_features, num_classes, latent_features) -> None:
        super().__init__()

        self.cnn_0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=in_features, out_channels=latent_features,
                kernel_size=5, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=latent_features)),
            ("activation", nn.ReLU()),
            ("pool", nn.MaxPool1d(kernel_size=2))
            ]))

        # convolutional layer 1
        self.cnn_1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features, out_channels=latent_features,
                kernel_size=3, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=latent_features)),
            ("activation", nn.ReLU()),
            ("pool", nn.MaxPool1d(kernel_size=2))
            ]))

        # convolutional layer 2
        self.cnn_2 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features, out_channels=latent_features*2,
                kernel_size=3, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=latent_features*2)),
            ("activation", nn.ReLU()),
            ("pool", nn.MaxPool1d(kernel_size=2))
            ]))

        self.last = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features*2, out_channels=latent_features*4,
                kernel_size=1, padding="same")),
            ("activation", nn.ReLU())
            ]))

        self.tcnn_0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features*4, out_channels=latent_features*2,
                kernel_size=3, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=latent_features*2)),
            ("activation", nn.ReLU()),
            ]))

        # convolutional layer 1
        self.tcnn_1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features*2, out_channels=latent_features,
                kernel_size=3, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=latent_features)),
            ("activation", nn.ReLU()),
            ]))

        self.tcnn_2 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features, out_channels=latent_features,
                kernel_size=5, padding="same")),
            ("activation", nn.ReLU())
            ]))

        self.out = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=latent_features, out_channels=num_classes,
                kernel_size=1, padding="same")),
            ("activation", nn.ReLU())
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_0(x)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.last(x)

        x = F.interpolate(x, size=(x.shape[-1]*2, ), mode='linear', align_corners=False)
        x = self.tcnn_0(x)
        x = F.interpolate(x, size=(x.shape[-1]*2, ), mode='linear', align_corners=False)
        x = self.tcnn_1(x)
        x = F.interpolate(x, size=(x.shape[-1]*2, ), mode='linear', align_corners=False)
        x = self.tcnn_2(x)
        return self.out(x)
