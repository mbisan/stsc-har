import torch
from torch import nn, Tensor

import numpy as np

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
        in_dimensions: int,
        latent_dims: int,
        dropout: float = 0,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        feedforward_mult: int = 2
    ):
        super().__init__()

        self.embedding = nn.Linear( # takes input of shape (n, t, d) -> (n, t, decoder_output_size)
            in_features=in_dimensions,
            out_features=latent_dims
        )

        self.positional_encoder = PositionalEncoding(latent_dims, dropout=dropout, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dims,
            nhead=n_heads,
            dim_feedforward=latent_dims * feedforward_mult,
            batch_first=True,
        ) # (batch, seq, feature) -> (batch, seq, feature)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )

    def forward(
        self,
        src: Tensor,
        mask: Tensor=None
    ):
        src = self.embedding(src) # (n, t, d) -> (n, t, latent)
        src = self.positional_encoder(src) # (n, t, latent) -> (n, t, latent)

        src = self.encoder(
            src=src,
            mask=mask
        ) # (n, t, latent) -> (n, t, latent)

        # reduce time dimension with global average pooling
        src = src.mean(dim=1) # (n, t, latent) -> (n, latent)

        return src

# channels, ref_size,
#             wdw_size, n_feature_maps

class Transformer(nn.Module):
    def __init__(self, channels, ref_size,
            wdw_size, n_feature_maps):

        super().__init__()

        self.channels = channels
        self.ref_size = ref_size
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

        self.transformer = TimeSeriesTransformer(
            in_dimensions=channels,
            latent_dims=n_feature_maps,
            dropout=0,
            n_heads=8,
            n_encoder_layers=4,
            feedforward_mult=2
        )

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape

    def forward(self, x):
        mask = generate_square_subsequent_mask(x.shape[-1], x.shape[-1]).to(x.device)
        x = x.permute((0, 2, 1))
        return self.transformer(x, mask=mask)
