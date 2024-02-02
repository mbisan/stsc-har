from nets.encoders.CNN_GAP_img import CNN_GAP_IMG
from nets.encoders.CNN_GAP_ts import CNN_GAP_TS
from nets.encoders.encoders import RemoveUpperPart, NoLayer
from nets.encoders.transformer import Transformer

from nets.decoders.mlp import MultiLayerPerceptron

encoder_dict = {
    "cnn_gap_img": CNN_GAP_IMG,
    "cnn_gap_ts": CNN_GAP_TS,
    "none": NoLayer,
    "noupper": RemoveUpperPart,
    "transformer": Transformer
}

decoder_dict = {
    "mlp": MultiLayerPerceptron
}