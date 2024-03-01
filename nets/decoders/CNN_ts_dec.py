# pylint: disable=invalid-name

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

class CNN_TS_dec(torch.nn.Module):

    def __init__(self, inp_feats: int, hid_feats: int, out_feats: int,
            _: int = 1 ):
        super().__init__()

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=inp_feats, out_channels=hid_feats*2, kernel_size=5, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=hid_feats*2)),
            ("activation", nn.ReLU()),
            ]))

        # convolutional layer 1
        self.cnn_1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=hid_feats*2, out_channels=hid_feats, kernel_size=3, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=hid_feats)),
            ("activation", nn.ReLU())
            ]))

        self.last = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=hid_feats, out_channels=out_feats, kernel_size=1, padding="same")),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_0(x)
        x = F.interpolate(x, (x.shape[-1]*2, ), mode="linear", align_corners=False)
        x = self.cnn_1(x)
        x = F.interpolate(x, (x.shape[-1]*2, ), mode="linear", align_corners=False)
        x = self.last(x)
        return x
