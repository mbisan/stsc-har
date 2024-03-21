# pylint: disable=invalid-name

from collections import OrderedDict

import torch
from torch import nn

class CNN_GAP_IMG3(torch.nn.Module):

    # pylint: disable=too-many-instance-attributes

    def __init__(self, channels=1, ref_size=32,
            wdw_size=32, n_feature_maps=32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = ref_size
        self.n_feature_maps = n_feature_maps

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels=channels, out_channels=self.n_feature_maps,
                kernel_size=5, padding='same')),
            ("bn", nn.BatchNorm2d(num_features=self.n_feature_maps)),
            ("activation", nn.ReLU()),
            ("pool", nn.MaxPool2d(kernel_size=2))
            ]))

        # convolutional layer 1
        self.cnn_1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels=self.n_feature_maps, out_channels=self.n_feature_maps,
                kernel_size=4, padding='same')),
            ("bn", nn.BatchNorm2d(num_features=self.n_feature_maps)),
            ("activation", nn.ReLU()),
            ("pool", nn.MaxPool2d(kernel_size=2))
            ]))

        # convolutional layer 2
        self.cnn_2 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels=self.n_feature_maps, out_channels=self.n_feature_maps*2,
                kernel_size=3, padding='valid')),
            ("bn", nn.BatchNorm2d(num_features=self.n_feature_maps*2)),
            ("activation", nn.ReLU())
            ]))

        self.last = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels=self.n_feature_maps*2, out_channels=self.n_feature_maps*4,
                kernel_size=1, padding="same")),
            ("activation", nn.ReLU())
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_0(x)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.last(x)
        return x.mean(dim=-2)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
