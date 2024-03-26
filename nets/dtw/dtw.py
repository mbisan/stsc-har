import torch

@torch.jit.script
def dtw_compute_full(
    dtw: torch.Tensor, dist_grad: torch.Tensor, dim: int, w: float) -> torch.Tensor:
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pattern_len)
    '''
    n, k, len_pattern, len_window = dtw.shape
    # very big tensor
    grads = torch.zeros(n, k, dim, len_pattern, len_pattern, len_window, device=dist_grad.device)
    # shape (n, k, dims, pattern_len, pattern_len, window_size)

    temp = torch.cumsum(dist_grad, dim=4)
    for i in range(len_pattern):
        grads[:, :, :, i, i, :] = temp[:, :, :, i, :]
        grads[:, :, :, i, i:, 0] = grads[:, :, :, i, i, :1]

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            temp_1 = dtw[:, :, i, j-1] < dtw[:, :, i-1, j-1] # path not (i-1, j-1)
            temp_2 = w * dtw[:, :, i, j-1] < dtw[:, :, i-1, j] # path not (i-1, j)

            pathijm1 = temp_1 & temp_2
            pathim1jm1 = (~temp_1) & temp_2
            pathim1j = temp_1 & (~temp_2)

            dtw[pathijm1][:, i, j] += w * dtw[pathijm1][:, i, j-1]
            dtw[pathim1jm1][:, i, j] += dtw[pathim1jm1][:, i-1, j]
            dtw[pathim1j][:, i, j] += w * dtw[pathim1j][:, i-1, j-1]

            grads[pathijm1][:, :, :, i, j] += w * grads[pathijm1][:, :, :, i, j-1]
            grads[pathim1jm1][:, :, :, i, j] += grads[pathim1jm1][:, :, :, i-1, j]
            grads[pathim1j][:, :, :, i, j] += w * grads[pathim1j][:, :, :, i-1, j-1]

    return grads

@torch.jit.script
def dtw_compute_no_grad(dtw: torch.Tensor, w: float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
        grad of shape (n, k, dims, pattern_len)
    '''

    _, _, len_pattern, len_window = dtw.shape

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(
                w * torch.minimum(dtw[:, :, i, j-1], dtw[:, :, i-1, j-1]),dtw[:, :, i-1, j])

            dtw[:, :, i, j] += value

@torch.jit.script
def dtw_fast_full(
    x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    if compute_gradients:
        p_diff /= torch.sqrt(euc_d[:,:, None, :, :] + eps)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    if compute_gradients:
        # p_diff now contains the partial derivatives of
        # DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))

        grads = dtw_compute_full(euc_d, p_diff, x.shape[1], w) # dims (n, k, d, i, i, j)

        return euc_d.sqrt(), grads

    dtw_compute_no_grad(euc_d, w)

    return euc_d.sqrt(), None

# pylint: disable=invalid-name abstract-method arguments-differ
class torch_dtw(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor, w: float):
        DTW, p_diff = dtw_fast_full(x, y, w, compute_gradients=y.requires_grad)
        return DTW, p_diff

    @staticmethod
    def setup_context(ctx, _, output):
        _, p_diff = output
        ctx.save_for_backward(p_diff)

    @staticmethod
    def backward(ctx, dtw_grad, _):
        # dtw_grad dims (n, k, i, j) p_diff dims (n, k, d, i, i, j)
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:, :, None, :, None, :]) # dims (n, k, d, i, i, j)
        return None, mult.sum(dim=(-2, -1)).mean(dim=0), None
