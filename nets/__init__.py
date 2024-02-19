from nets.encoders.CNN_GAP_img import CNN_GAP_IMG
from nets.encoders.CNN_GAP_ts import CNN_GAP_TS
from nets.encoders.encoders import RemoveUpperPart, NoLayer
from nets.encoders.transformer import Transformer

encoder_dict = {
    "cnn_gap_img": CNN_GAP_IMG,
    "cnn_gap_ts": CNN_GAP_TS,
    "none": NoLayer,
    "noupper": RemoveUpperPart,
    "transformer": Transformer
}

from nets.decoders.mlp import MultiLayerPerceptron

decoder_dict = {
    "mlp": MultiLayerPerceptron
}

from nets.segmentation.deeplabv3p1d import get_model
from nets.segmentation.unet import UNET
from nets.segmentation.utime import UTime

segmentation_dict = {
    "dlv3": get_model,
    "unet": UNET,
    "utime": UTime
}