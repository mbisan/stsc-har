# code modified from https://github.com/VainF/DeepLabV3Plus-Pytorch

'''
MIT License

Copyright (c) 2020 Gongfan Fang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# pylint: disable=invalid-name

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

class CNN_TS(torch.nn.Module):

    def __init__(self, channels=1, _=32,
            wdw_size=32, n_feature_maps=32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=channels, out_channels=self.n_feature_maps,
                kernel_size=5, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=self.n_feature_maps)),
            ("activation", nn.ReLU())
            ]))

        # convolutional layer 1
        self.cnn_1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=self.n_feature_maps, out_channels=self.n_feature_maps,
                kernel_size=4, padding='same')),
            ("bn", nn.BatchNorm1d(num_features=self.n_feature_maps)),
            ("activation", nn.ReLU())
            ]))

        # convolutional layer 2
        self.cnn_2 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=self.n_feature_maps, out_channels=self.n_feature_maps*2,
                kernel_size=3, padding='valid')),
            ("bn", nn.BatchNorm1d(num_features=self.n_feature_maps*2)),
            ("activation", nn.ReLU())
            ]))

        self.last = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(
                in_channels=self.n_feature_maps*2, out_channels=self.n_feature_maps*4,
                kernel_size=1, padding="same")),
            ("activation", nn.ReLU())
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_0(x)
        out = self.cnn_1(x)
        out = self.cnn_2(out)
        out = self.last(out)
        return {"out": out, "low_level": x} # global average pooling

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-1:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='linear', align_corners=False)
        return x

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

class DeepLabHeadV3Plus1D(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=(12, 24, 36)):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(low_level_channels, low_level_channels, 1, bias=False),
            nn.BatchNorm1d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP1D(in_channels, aspp_dilate, out_channels=in_channels)

        self.classifier = nn.Sequential(
            nn.Conv1d(
                in_channels+low_level_channels,
                (in_channels+low_level_channels), 3, padding=1, bias=False),
            nn.BatchNorm1d((in_channels+low_level_channels)),
            nn.ReLU(inplace=True),
            nn.Conv1d((in_channels+low_level_channels), num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(
            output_feature, size=low_level_feature.shape[-1:],
            mode='linear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv1D(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv1d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-1:]
        # pylint: disable=not-callable
        x = F.adaptive_avg_pool1d(x, int(size[0]/4))
        x = self.layers(x)
        return F.interpolate(x, size=size, mode='linear', align_corners=False)

class ASPP1D(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels = 256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)))

        for rate in tuple(atrous_rates):
            modules.append(ASPPConv1D(in_channels, out_channels, rate))

        modules.append(ASPPPooling1D(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        n_convs = 2 + len(atrous_rates)

        self.project = nn.Sequential(
            nn.Conv1d(n_convs * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def get_model(in_channels, latent_features, n_classes, aspp_dilate=(2, 4)):
    return DeepLabV3(
        CNN_TS(channels=in_channels, n_feature_maps=latent_features),
        DeepLabHeadV3Plus1D(4*latent_features, latent_features, n_classes, aspp_dilate=aspp_dilate))
