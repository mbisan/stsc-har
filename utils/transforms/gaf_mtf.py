from typing import Tuple

import torch
from torch import nn

# Gramian Frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# pylint: disable=invalid-name

@torch.jit.script
def minmax_scaler(
    X: torch.Tensor, minmax: Tuple[float, float] = (-1, 1), eps: float = 1e-6) -> torch.Tensor:
    '''
        Scales the last dimension of X into the given range X has shape (...,d)
        output has the same shape but the last dimension is scaled to the range
    '''
    X_min = X.min(dim=-1, keepdim=True).values
    X_std = (X - X_min)/(X.max(dim=-1, keepdim=True).values - X_min + eps)
    return X_std * (minmax[1] - minmax[0]) + minmax[0]

@torch.jit.script
def gaf_compute(
    X: torch.Tensor, mode: str = "s", scaling: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    '''
        Computes the Gramian Angular Field of time series X, per sample, per channel
        input has shape (...,N, d, window_size) (any number of leading dimensions)
        output has shape (..., N, d, window_size, window_size)

        First scales the window_size dimension to -1, 1 range and then computes the GAF
        either summation (setting mode to "s" or "summation") or difference, which is the default
        behaviour.
    '''
    X_cos = minmax_scaler(X, scaling)
    X_sin = torch.clamp((1.0 - X_cos.square()).sqrt(), 0, 1)

    if "s" in mode: # summation
        return X_cos.unsqueeze(-2) * X_cos.unsqueeze(-1) - X_sin.unsqueeze(-2) * X_sin.unsqueeze(-1)
    # difference
    return X_sin.unsqueeze(-2) * X_cos.unsqueeze(-1) - X_cos.unsqueeze(-2) * X_sin.unsqueeze(-1)

# Markov transition Fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@torch.jit.script
def mtm_compute(X_binned: torch.Tensor, n_bins: int) -> torch.Tensor:
    # compute markov transition matrix
    current_bin = X_binned[..., :-1].view(-1, X_binned.shape[-1]-1)
    next_bin = X_binned[..., 1:].view(-1, X_binned.shape[-1]-1)

    # indices to sum +1
    indices = current_bin * n_bins + next_bin

    X_mtm = torch.zeros(X_binned.shape[:-1] + (n_bins, n_bins), device=X_binned.device)
    X_mtm.view(-1, n_bins*n_bins).scatter_add_(
        1, indices, torch.ones_like(indices, dtype=X_mtm.dtype))

    sum_mtm = X_mtm.sum(dim=-1)
    X_mtm /= torch.where(sum_mtm[..., None] == 0, 1, sum_mtm[..., None]) # normalize

    return X_mtm

@torch.jit.script
def mtf_compute(
    X: torch.Tensor, bins: int = 50, scaling: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    '''
        Computes the Markov Transition Field of time series X, per sample, per channel
        input has shape (...,N, d, window_size) (any number of leading dimensions)
        output has shape (..., N, d, window_size, window_size)

        Bins are uniformly spaced then computes the MTF of size (window_size * window_size)

        NOTE This implementation is the same as the one in tslearn for uniform bins
    '''

    X_scaled = minmax_scaler(X, scaling)
    wsize = X.shape[-1]
    bins_boundaries = torch.linspace(scaling[0], scaling[1], bins+1, device=X.device)[1:-1]

    X_binned = torch.bucketize(X_scaled, bins_boundaries)
    n_bins = bins_boundaries.numel() + 1

    X_mtm = mtm_compute(X_binned, n_bins) # (..., N, d, bins, bins)

    # X_mtf[i, j, k] = X_mtm[i, X_binned[i, j], X_binned[i, k]]
    X_binned_flat = X_binned.view(-1, wsize)
    indices = n_bins * X_binned_flat[:, :, None] + X_binned_flat[:, None, :]
    X_mtf_flat = X_mtm.view(-1, n_bins * n_bins)
    X_mtf = torch.gather(X_mtf_flat, 1, indices.view(indices.size(0), -1)).view(X.shape + (wsize,))

    return X_mtf

class GAFLayer(nn.Module):

    def __init__(self, mode, scaling=(-1, 1)) -> None:
        super().__init__()

        self.mode = mode
        self.scaling = scaling

    def forward(self, X):
        return gaf_compute(X, self.mode, self.scaling)

class MTFLayer(nn.Module):

    def __init__(self, bins, scaling=(-1, 1)) -> None:
        super().__init__()

        self.bins = bins
        self.scaling = scaling

    def forward(self, X):
        return mtf_compute(X, self.bins, self.scaling)
