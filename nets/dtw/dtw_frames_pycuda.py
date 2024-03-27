import torch
import torch.cuda
from numba import cuda

from nets.dtw.dtw_pycuda import dtw_fill_first_line

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
                        grads[x, y, l, i, j] += w * grads[x, y, l, i-1, j]

        cuda.syncthreads()

# @torch.jit.script
def dtw_forward(x: torch.Tensor, y: torch.Tensor, w: float):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    n, k, dim, len_pattern, len_window = p_diff.shape

    # if compute_gradients:
    #     p_diff /= euc_d[:,:, None, :, :] + eps

    # compute dtw
    dtw_fill_first_line(euc_d, w)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    grads = torch.zeros(n, k, dim, len_pattern, len_pattern, len_window, device=x.device)

    if y.requires_grad:
        dtw_fill_with_grads[(16, 16), (16, 16)](
            cuda.as_cuda_array(euc_d), cuda.as_cuda_array(grads), w)
    else:
        dtw_fill[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w)

    return euc_d, p_diff
