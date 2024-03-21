from nets.encoders.CNN_GAP_img import CNN_GAP_IMG
from nets.encoders.CNN_GAP_img2 import CNN_GAP_IMG2
from nets.encoders.CNN_GAP_img3 import CNN_GAP_IMG3
from nets.encoders.CNN_img import CNN_IMG
from nets.encoders.CNN_GAP_ts import CNN_GAP_TS
from nets.encoders.CNN_ts import CNN_TS
from nets.encoders.CNN_2d_ts import CNN_2d_TS
from nets.encoders.encoders import RemoveUpperPart, NoLayer
from nets.encoders.transformer import Transformer
from nets.encoders.LSTM import RNN_ts

from nets.decoders.mlp import MultiLayerPerceptron
from nets.decoders.CNN_ts_dec import CNN_TS_dec

from nets.segmentation.deeplabv3p1d import get_model
from nets.segmentation.unet import UNET
from nets.segmentation.utime import UTime

encoder_dict = {
    "cnn_gap_img": CNN_GAP_IMG,
    "cnn_gap_img2": CNN_GAP_IMG2,
    "cnn_gap_img3": CNN_GAP_IMG3,
    "cnn_img": CNN_IMG,
    "cnn_gap_ts": CNN_GAP_TS,
    "cnn_ts": CNN_TS,
    "none": NoLayer,
    "noupper": RemoveUpperPart,
    "transformer": Transformer,
    "cnn_2d_ts": CNN_2d_TS,
    "lstm": RNN_ts
}

decoder_dict = {
    "mlp": MultiLayerPerceptron,
    "cnn_ts_dec": CNN_TS_dec
}

segmentation_dict = {
    "dlv3": get_model,
    "unet": UNET,
    "utime": UTime
}
