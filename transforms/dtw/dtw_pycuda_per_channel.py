import numpy as np
import torch
import torch.cuda
from numba import cuda

@cuda.jit
def dtw_fill_c(dtw, w):
    '''
        dtw of shape (n, k, d, pattern_len, window_size)
    '''
    n, k, d, len_pattern, len_window = dtw.shape

    x, y, h = cuda.grid(3)

    if x < n and y < k and h < d:
        for i in range(1, len_pattern): # pl
            for j in range(1, len_window): # ws
                value = min(w * min(dtw[x, y, h, i, j-1], dtw[x, y, h, i-1, j-1]), dtw[x, y, h, i-1, j])
                dtw[x, y, h, i, j] += value

        cuda.syncthreads()

@cuda.jit
def dtw_backward_c(dtw, dist_grad, grad):
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pl)
    '''
    n, k, d, len_pattern, len_window = dist_grad.shape

    x, y, h = cuda.grid(3)

    if x < n and y < k and h < d:
        for i0 in range(len_pattern-1, -1, -1):
            for j0 in range(len_window-1, -1, -1):

                # A = dtw[x, y, h, i0, j0-1]
                # B = dtw[x, y, h, i0-1, j0]
                # C = dtw[x, y, h, i0-1, j0-1]

                # path is A if (A<B) & (A<C) -> path is not A if (A>=B) | (A>=C)
                # path is B if (B<A) & (B<C) -> path is not B if (B>=A) | (B>=C)

                if dtw[x, y, h, i0, j0] != np.inf:

                    grad[x, y, h, i0] += dist_grad[x, y, h, i0, j0]
            
                    if j0==0 or i0==0:
                        continue

                    if dtw[x, y, h, i0, j0-1] >= dtw[x, y, h, i0-1, j0] or dtw[x, y, h, i0, j0-1] >= dtw[x, y, h, i0-1, j0-1]: # path is not A
                        for j in range(j0):
                            dtw[x, y, h, i0, j] = np.inf
                    if dtw[x, y, h, i0-1, j0] >= dtw[x, y, h, i0, j0-1] or dtw[x, y, h, i0-1, j0] >= dtw[x, y, h, i0-1, j0-1]: # path is not B
                        for i in range(i0):
                            dtw[x, y, h, i, j0] = np.inf

        cuda.syncthreads()

# @torch.jit.script
def dtw_forward_c(x: torch.Tensor, y: torch.Tensor, w: float):
    # shape of x (n, dim, x_len) y (m, y_len)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,None,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.abs(p_diff) # shape (n, n_kernel, d, kernel_size, T)

    # compute dtw
    euc_d[:,:,:,0,:] = torch.cumsum(euc_d[:,:,:,0,:], dim=2)
    euc_d[:,:,:,:,0] = torch.cumsum(euc_d[:,:,:,:,0], dim=2)

    dtw_fill_c[(16, 4, 4), (16, 4, 4)](cuda.as_cuda_array(euc_d), w)

    return euc_d, torch.where(p_diff < 0, 1.0, -1.0)
    
class torch_dtw_cuda_c(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, w: float = 1):
        DTW, p_diff = dtw_forward_c(x, y, w)

        ctx.save_for_backward(DTW, p_diff)

        return DTW[:, :, :, -1, -1]
    
    @staticmethod
    def backward(ctx, dtw_grad):
        # dtw_grad dims (n, k, d)
        dtw, p_diff = ctx.saved_tensors
        grads = torch.zeros((dtw.shape[0],) + p_diff.shape[1:-1], device=dtw_grad.device)
        dtw_backward_c[(16, 4, 4), (16, 4, 4)](cuda.as_cuda_array(dtw), cuda.as_cuda_array(p_diff), cuda.as_cuda_array(grads))

        mult = (dtw_grad[:, :, :, None] * grads) # dims (n, k, d)
        return None, mult.mean(dim=(0, 2)), None # dims (n, d, k)