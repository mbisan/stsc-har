"""
Implementation of UTime as described in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)

Adapted for pytorch from https://github.com/perslev/U-Time/
"""

from typing import Tuple
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class UTimeDecoder(nn.Module):

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        res_channels,
        pools,
        depth,
        filters,
        kernel_size,
        padding,
        complexity_factor
    ):
        super().__init__()

        self.depth = depth
        self.pools = pools

        for i in range(depth):
            filters = int(filters/2)
            setattr(self, f"decoder_0_{i}", nn.Sequential(OrderedDict([
                ("conv", nn.Conv1d(
                    res_channels[-(i+1)], int(filters*complexity_factor),
                    kernel_size=kernel_size,
                    padding=padding)),
                ("activation", nn.ReLU()),
                ("bn", nn.BatchNorm1d(int(filters*complexity_factor))),
            ])))

            setattr(self, f"decoder_1_{i}", nn.Sequential(OrderedDict([
                ("conv0", nn.Conv1d(
                    res_channels[-(i+1)] + int(filters*complexity_factor),
                    int(filters*complexity_factor), kernel_size=kernel_size,
                    padding=padding)),
                ("activation0", nn.ReLU()),
                ("bn0", nn.BatchNorm1d(int(filters*complexity_factor))),
                ("conv1", nn.Conv1d(
                    int(filters*complexity_factor),
                    int(filters*complexity_factor), kernel_size=kernel_size,
                    padding=padding)),
                ("activation1", nn.ReLU()),
                ("bn1", nn.BatchNorm1d(int(filters*complexity_factor))),
            ])))

        self.filters = filters

    def forward(self, x, residual_conn):

        for i in range(self.depth):
            x = F.interpolate(x, scale_factor=self.pools[-(i+1)], mode="linear")
            x = getattr(self, f"decoder_0_{i}")(x)
            x = torch.cat([residual_conn[-(i+1)], x], dim=1)
            x = getattr(self, f"decoder_1_{i}")(x)

        return x

class UTimeEncoder(nn.Module):

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        in_channels,
        depth,
        pools,
        filters,
        kernel_size,
        dilation,
        padding,
        complexity_factor
    ):
        super().__init__()

        setattr(self, "encoder0", nn.Sequential(OrderedDict([
            ("conv0", nn.Conv1d(
                in_channels, int(filters*complexity_factor), kernel_size=kernel_size,
                padding=padding, dilation=dilation)),
            ("activation0", nn.ReLU()),
            ("bn0", nn.BatchNorm1d(int(filters*complexity_factor))),
            ("conv1", nn.Conv1d(
                int(filters*complexity_factor), int(filters*complexity_factor),
                kernel_size=kernel_size,
                padding=padding, dilation=dilation)),
            ("activation1", nn.ReLU()),
            ("bn1", nn.BatchNorm1d(int(filters*complexity_factor))),
        ])))

        self.pools = pools
        self.depth = depth
        self.res_channels = [int(filters*complexity_factor)]

        for i in range(1, depth):
            filters *= 2
            setattr(self, f"encoder{i}", nn.Sequential(OrderedDict([
                ("conv0", nn.Conv1d(
                    self.res_channels[-1], int(filters*complexity_factor),
                    kernel_size=kernel_size,
                    padding=padding, dilation=dilation)),
                ("activation0", nn.ReLU()),
                ("bn0", nn.BatchNorm1d(int(filters*complexity_factor))),
                ("conv1", nn.Conv1d(
                    int(filters*complexity_factor), int(filters*complexity_factor),
                    kernel_size=kernel_size,
                    padding=padding, dilation=dilation)),
                ("activation1", nn.ReLU()),
                ("bn1", nn.BatchNorm1d(int(filters*complexity_factor))),
            ])))
            self.res_channels += [int(filters*complexity_factor)]

        self.filters = filters

    def forward(self, x):
        out = [None for i in range(self.depth)]
        out[0] = self.encoder0(x)

        for i in range(1, self.depth):
            out[i] = getattr(self, f"encoder{i}")(F.max_pool1d(out[i-1], self.pools[i-1]))

        return F.max_pool1d(out[-1], kernel_size=self.pools[-1]), out

class UTime(nn.Module):
    """
    See also original U-net paper at http://arxiv.org/abs/1505.04597
    """

    # pylint: disable=too-many-arguments too-many-instance-attributes

    def __init__(self,
        n_classes: int,
        in_dims: int,
        depth: int = 4,
        dilation: tuple = 2,
        kernel_size: int = 5,
        padding: str = "same",
        init_filters: int = 16,
        complexity_factor: int = 2,
        pools: Tuple[int] = (10, 8, 6, 4),
        segment_size: int = 1,
        change_size: int = 1,
        **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        batch_shape (list): Giving the shape of one one batch of data,
                            potentially omitting the zeroth axis (the batch
                            size dim)
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        dilation (int):
            TODO
        activation (string):
            Activation function for convolution layers
        dense_classifier_activation (string):
            TODO
        kernel_size (int):
            Kernel size for convolution layers
        transition_window (int):
            TODO
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            convolution layer instead of default N.
        l2_reg (float in [0, 1])
            L2 regularization on conv weights
        pools (int or list of ints):
            TODO
        data_per_prediction (int):
            TODO
        build (bool):
            TODO
        """
        super().__init__()

        # Set various attributes

        self.input_dims = in_dims
        self.n_classes = n_classes

        self.dilation = dilation
        self.cf = np.sqrt(complexity_factor)
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.n_crops = 0

        self.pools = pools
        if len(self.pools) != self.depth:
            raise ValueError("Argument 'pools' must be a single integer or a "
                             "list of values of length equal to 'depth'.")

        self.padding = padding.lower()
        if self.padding != "same":
            raise ValueError("Currently, must use 'same' padding.")

        self.encoder = UTimeEncoder(
            self.input_dims,
            self.depth,
            self.pools,
            self.init_filters,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.cf
        )

        self.decoder = UTimeDecoder(
            self.encoder.res_channels,
            self.pools,
            self.depth,
            self.encoder.filters,
            self.kernel_size,
            self.padding,
            self.cf
        )

        self.project = nn.Conv1d(
            in_channels=int(self.decoder.filters*self.cf), out_channels=self.n_classes,
            kernel_size=1, padding="same")

        self.change_size = change_size
        self.segment_size = segment_size

        self.project2 = nn.Conv1d(
            in_channels=self.n_classes, out_channels=self.n_classes,
            kernel_size=change_size, padding="same")

    def forward(self, x):
        x, low_feats = self.encoder(x)
        x = self.decoder(x, low_feats)
        x = self.project(x)
        # pylint: disable=not-callable
        x = F.avg_pool1d(x, kernel_size=self.segment_size)
        return self.project2(x).repeat_interleave(self.segment_size, dim=-1)
