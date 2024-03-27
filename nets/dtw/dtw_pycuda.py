import numpy as np
import torch
import torch.cuda
from numba import cuda

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
def dtw_backward(dtw, dist_grad, grad):
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pl)
    '''
    # pylint: disable=unbalanced-tuple-unpacking no-value-for-parameter comparison-with-callable

    n, k, d, len_pattern, len_window = dist_grad.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for i0 in range(len_pattern-1, -1, -1):
            for j0 in range(len_window-1, -1, -1):

                # A = dtw[x, y, i0, j0-1]
                # B = dtw[x, y, i0-1, j0]
                # C = dtw[x, y, i0-1, j0-1]

                # path is A if (A<B) & (A<C) -> path is not A if (A>=B) | (A>=C)
                # path is B if (B<A) & (B<C) -> path is not B if (B>=A) | (B>=C)

                if dtw[x, y, i0, j0] != np.inf:

                    for l in range(d):
                        # pylint: disable=too-many-function-args
                        cuda.atomic.add(grad, (x, y, l, i0), dist_grad[x, y, l, i0, j0])

                    if j0==0 or i0==0:
                        continue

                    if dtw[x, y, i0, j0-1] >= dtw[x, y, i0-1, j0] or \
                       dtw[x, y, i0, j0-1] >= dtw[x, y, i0-1, j0-1]: # path is not A
                        for j in range(j0):
                            dtw[x, y, i0, j] = np.inf
                    if dtw[x, y, i0-1, j0] >= dtw[x, y, i0, j0-1] or \
                       dtw[x, y, i0-1, j0] >= dtw[x, y, i0-1, j0-1]: # path is not B
                        for i in range(i0):
                            dtw[x, y, i, j0] = np.inf

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

    # if compute_gradients:
    #     p_diff /= euc_d[:,:, None, :, :] + eps

    # compute dtw
    dtw_fill_first_line(euc_d, w)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    dtw_fill[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w)

    return euc_d, p_diff

# pylint: disable=invalid-name abstract-method arguments-differ
class torch_dtw_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, w: float = 1):
        DTW, p_diff = dtw_forward(x, y, w)

        ctx.save_for_backward(DTW, p_diff)

        return DTW[:, :, -1, -1]

    @staticmethod
    def backward(ctx, dtw_grad):
        # dtw_grad dims (n, k) p_diff dims (n, k, d, pl)
        dtw, p_diff = ctx.saved_tensors
        grads = torch.zeros((dtw.shape[0],) + p_diff.shape[1:-1], device=dtw_grad.device)
        dtw_backward[(16, 16), (16, 16)](
            cuda.as_cuda_array(dtw), cuda.as_cuda_array(p_diff), cuda.as_cuda_array(grads))

        mult = (dtw_grad[:, :, None, None] * grads) # dims (n, k, d)
        return None, mult.mean(0), None # dims (n, d, k)
