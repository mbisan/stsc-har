# pylint: disable=invalid-name

import torch
from torch import nn

class RNN_ts(nn.Module):

    def __init__(self, channels=1, latent_size=32,
            n_layers=2) -> None:
        super().__init__()

        self.channels = channels

        self.rnn = nn.LSTM(
            input_size=channels,
            hidden_size=latent_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None):
        # x of shape (n, c, ws)
        x = x.permute((0, 2, 1))
        # x of shape (n, ws, c)

        out, (h_n, _) = self.rnn(x, h_0) # h_n of shape (n_layers, n, latent_size)

        return out, h_n

    def get_output_shape(self):
        x = torch.rand((1, self.channels, 1))
        print("Input shape: ", x.shape)
        out, x = self(x)
        print("Latent shape: ", out.shape, x.shape)
        return x.shape
