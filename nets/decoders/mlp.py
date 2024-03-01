# pylint: disable=invalid-name

from torch import nn

class MultiLayerPerceptron(nn.Module):

    """ Multiplayer perceptron with ReLU activations """

    def __init__(self, inp_feats: int, hid_feats: int, out_feats: int,
            hid_layers: int = 1 ) -> None:

        super().__init__()

        assert hid_layers >= 0

        self.hid_layers = hid_layers
        self.features = [inp_feats] + [hid_feats for i in range(hid_layers)] + [out_feats]

        for hl in range(hid_layers):
            setattr(self, f"fcn_layer_{hl}",
                    nn.Linear(in_features=self.features[hl], out_features=self.features[hl+1]))
            setattr(self, f"act_layer_{hl}", nn.ReLU())
        self.fcn_out = nn.Linear(
            in_features=self.features[hid_layers], out_features=self.features[-1])

    def forward(self, x):
        for hl in range(self.hid_layers):
            x = getattr(self, f"fcn_layer_{hl}")(x)
            x = getattr(self, f"act_layer_{hl}")(x)
        x = self.fcn_out(x)
        return x
