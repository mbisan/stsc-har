from nets.encoders.CNN_GAP_img import CNN_GAP_IMG
from nets.encoders.CNN_GAP_ts import CNN_GAP_TS
from nets.encoders.CNN_ts import CNN_TS
from nets.encoders.encoders import RemoveUpperPart, NoLayer
from nets.encoders.transformer import Transformer

encoder_dict = {
    "cnn_gap_img": CNN_GAP_IMG,
    "cnn_gap_ts": CNN_GAP_TS,
    "cnn_ts": CNN_TS,
    "none": NoLayer,
    "noupper": RemoveUpperPart,
    "transformer": Transformer
}

from nets.decoders.mlp import MultiLayerPerceptron
from nets.decoders.CNN_ts_dec import CNN_TS_dec

decoder_dict = {
    "mlp": MultiLayerPerceptron,
    "cnn_ts_dec": CNN_TS_dec
}

from nets.segmentation.deeplabv3p1d import get_model
from nets.segmentation.unet import UNET
from nets.segmentation.utime import UTime

segmentation_dict = {
    "dlv3": get_model,
    "unet": UNET,
    "utime": UTime
}