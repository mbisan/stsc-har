import numpy as np
import torch
import torch.cuda
from numba import cuda, jit, prange

@jit(nopython=True, parallel=True)
def dtw_fill_first_line_cpu(dtw: np.ndarray, w:float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=not-an-iterable

    n, k, _, len_window = dtw.shape

    for x in prange(n):
        for y in prange(k):
            for j in range(1, len_window): # ws
                dtw[x, y, 0, j] += w * dtw[x, y, 0, j-1]

@jit(nopython=True, parallel=True)
def dtw_fill_cpu(dtw: np.ndarray, w: float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=not-an-iterable

    n, k, len_pattern, len_window = dtw.shape

    for x in prange(n):
        for y in prange(k):
            for i in range(1, len_pattern): # pl
                for j in range(1, len_window): # ws
                    value = min(w * min(dtw[x, y, i, j-1], dtw[x, y, i-1, j-1]), dtw[x, y, i-1, j])
                    dtw[x, y, i, j] += value

@jit(nopython=True, parallel=True)
def dtw_fill_with_grads_cpu(dtw: np.ndarray, grads: np.ndarray, w: float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=not-an-iterable

    n, k, d, len_pattern, len_window = grads.shape

    for x in prange(n):
        for y in prange(k):
            for i in range(1, len_pattern): # pl
                for j in range(1, len_window): # ws
                    min_index, min_val = 0, dtw[x, y, i, j-1]
                    if dtw[x, y, i-1, j-1] < min_val:
                        min_index, min_val = 1, dtw[x, y, i-1, j-1]
                    if dtw[x, y, i-1, j] < w * min_val:
                        min_index, min_val = 2, dtw[x, y, i-1, j]

                    if min_index == 0:
                        dtw[x, y, i, j] += w * min_val
                        for l in range(d):
                            grads[x, y, l, i, j] += w * grads[x, y, l, i, j-1]

                    if min_index == 1:
                        dtw[x, y, i, j] += w * min_val
                        for l in range(d):
                            grads[x, y, l, i, j] += w * grads[x, y, l, i-1, j-1]

                    if min_index == 2:
                        dtw[x, y, i, j] += min_val
                        for l in range(d):
                            grads[x, y, l, i, j] += grads[x, y, l, i-1, j]

########### CUDA ###########

@cuda.jit
def dtw_fill_first_line(dtw: torch.Tensor, w:float):
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=unbalanced-tuple-unpacking no-value-for-parameter comparison-with-callable

    n, k, _, len_window = dtw.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for j in range(1, len_window): # ws
            dtw[x, y, 0, j] += w * dtw[x, y, 0, j-1]

        cuda.syncthreads()

@cuda.jit
def dtw_fill(dtw, w):
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=unbalanced-tuple-unpacking no-value-for-parameter comparison-with-callable

    n, k, len_pattern, len_window = dtw.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for i in range(1, len_pattern): # pl
            for j in range(1, len_window): # ws
                value = min(w * min(dtw[x, y, i, j-1], dtw[x, y, i-1, j-1]), dtw[x, y, i-1, j])
                dtw[x, y, i, j] += value

        cuda.syncthreads()

@cuda.jit
def dtw_fill_with_grads(dtw, grads, w):
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    # pylint: disable=unbalanced-tuple-unpacking no-value-for-parameter comparison-with-callable

    n, k, d, len_pattern, len_window = grads.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for i in range(1, len_pattern): # pl
            for j in range(1, len_window): # ws
                min_index, min_val = 0, dtw[x, y, i, j-1]
                if dtw[x, y, i-1, j-1] < min_val:
                    min_index, min_val = 1, dtw[x, y, i-1, j-1]
                if dtw[x, y, i-1, j] < w * min_val:
                    min_index, min_val = 2, dtw[x, y, i-1, j]

                if min_index == 0:
                    dtw[x, y, i, j] += w * min_val
                    for l in range(d):
                        grads[x, y, l, i, j] += w * grads[x, y, l, i, j-1]

                if min_index == 1:
                    dtw[x, y, i, j] += w * min_val
                    for l in range(d):
                        grads[x, y, l, i, j] += w * grads[x, y, l, i-1, j-1]

                if min_index == 2:
                    dtw[x, y, i, j] += min_val
                    for l in range(d):
                        grads[x, y, l, i, j] += grads[x, y, l, i-1, j]

        cuda.syncthreads()

########### TORCH ###########

# @torch.jit.script
def dtw_forward(x: torch.Tensor, y: torch.Tensor, w: float, compute_grads: bool):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    # if compute_gradients:
    #     p_diff /= euc_d[:,:, None, :, :] + eps

    # compute dtw
    if x.is_cuda:
        dtw_fill_first_line[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w)
        euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

        if compute_grads:
            p_diff *= 2.0
            dtw_fill_with_grads[(16, 16), (16, 16)](
                cuda.as_cuda_array(euc_d), cuda.as_cuda_array(p_diff), w)
        else:
            dtw_fill[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w)

        return euc_d, p_diff
    # else
    dtw_fill_first_line_cpu(euc_d.numpy(), w)

    if compute_grads:
        p_diff *= 2.0
        dtw_fill_with_grads_cpu(
            euc_d.numpy(), p_diff.numpy(), w)
    else:
        dtw_fill_cpu(euc_d.numpy(), w)

    return euc_d, p_diff

# pylint: disable=invalid-name abstract-method arguments-differ
class torch_dtw_frames_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, w: float = 1, compute_grads: bool = True):
        DTW, p_diff = dtw_forward(x.detach(), y.detach(), w, compute_grads)

        ctx.save_for_backward(p_diff)

        return DTW # (n, k, pl, wl)

    @staticmethod
    def backward(ctx, dtw_grad):
        # dtw_grad dims (n, k, pl, wl) p_diff dims (n, k, d, pl, wl)
        p_diff, = ctx.saved_tensors

        mult = (dtw_grad[:, :, None, :, :] * p_diff) # dims (n, k, d, pl, wl)
        return None, mult.mean(0).mean(-1), None, None # dims (k, d, pl)

class DTWFramesLayerCUDA(torch.nn.Module):
    def __init__(self, n_patts, d_patts, l_patts, l_out: int = None, rho: float = 1) -> None:
        # pylint: disable=too-many-arguments
        super().__init__()

        if l_out is None:
            self.l_out = l_patts
        else:
            self.l_out = l_out

        self.w: torch.float32 = rho ** (1/l_patts)
        self.patts = torch.nn.Parameter(torch.randn(n_patts, d_patts, l_patts))

    def forward(self, x):
        x = torch_dtw_frames_cuda.apply(x, self.patts, self.w, self.training)[:,:,:,-self.l_out:]
        x = x.sqrt()/self.patts.shape[-1]
        return x
