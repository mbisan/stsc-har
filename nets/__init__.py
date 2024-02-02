from encoders.CNN_GAP_img import CNN_GAP_IMG
from encoders.CNN_GAP_ts import CNN_GAP_TS
from encoders.encoders import RemoveUpperPart, NoLayer
from encoders.transformer import Transformer

from decoders.mlp import MultiLayerPerceptron

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