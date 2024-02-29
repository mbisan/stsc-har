import torch

@torch.jit.script
def dtw_compute_full_per_channel(
    dtw: torch.Tensor, dist_grad: torch.Tensor, dim: int, w: float) -> torch.Tensor:
    '''
        dtw of shape (n, k, dims, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
    '''
    # pylint: disable=too-many-locals

    n, k, _, len_pattern, len_window = dtw.shape
    # very big tensor
    grads = torch.zeros(n, k, dim, len_pattern, len_pattern, len_window, device=dist_grad.device)
    # shape (n, k, dims, pattern_len, pattern_len, window_size)

    temp = torch.cumsum(dist_grad, dim=4)
    for i in range(len_pattern):
        grads[:, :, :, i, i, :] = temp[:, :, :, i, :]
        grads[:, :, :, i, i:, 0] = grads[:, :, :, i, i, :1]

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(
                dtw[:, :, :, i, j-1], dtw[:, :, :, i-1, j-1]), dtw[:, :, :, i-1, j])
            temp_1 = dtw[:, :, :, i, j-1] < dtw[:, :, :, i-1, j-1] # path (i, j-1) or (i-1, j)
            temp_2 = w * dtw[:, :, :, i, j-1] < dtw[:, :, :, i-1, j] # path (i, j-1) or (i-1, j-1)
            temp_3 = w * dtw[:, :, :, i-1, j-1] < dtw[:, :, :, i-1, j] # path (i-1, j-1) or (i-1, j)

            dtw[:, :, :, i, j] += value

            grads[temp_1 & temp_2][:i, i, j] += w * grads[temp_1 & temp_2][:i, i, j-1]
            grads[temp_1 & temp_3][:i, i, j] += grads[temp_1 & temp_3][:i, i-1, j]
            grads[temp_2 & temp_3][:i, i, j] += w * grads[temp_2 & temp_3][:i, i-1, j-1]

    return grads

@torch.jit.script
def dtw_compute_no_grad_per_channel(dtw: torch.Tensor, w: float) -> None:
    '''
        dtw of shape (n, k, d, pattern_len, window_size)
        grad of shape (n, k, d, dims, pattern_len)
    '''

    _, _, _, len_pattern, len_window = dtw.shape

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(
                dtw[:, :, :, i, j-1], dtw[:, :, :, i-1, j-1]), dtw[:, :, :, i-1, j])

            dtw[:, :, :, i, j] += value

# pylint: disable=unused-argument
@torch.jit.script
def dtw_fast_full_per_channel(
    x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True):
    # shape of x (n, dim, x_len) y (m, y_len)

    # performs convolution-like operation, for each kernel, for each dim the DF
    # (of shape (kernel_size, T)) is computed
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,None,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.abs(p_diff) # shape (n, n_kernel, d, kernel_size, T)

    # compute dtw
    euc_d[:,:,:,0,:] = torch.cumsum(euc_d[:,:,:,0,:], dim=3)
    euc_d[:,:,:,:,0] = torch.cumsum(euc_d[:,:,:,:,0], dim=3)

    if compute_gradients:
        # p_diff now contains the partial derivatives of
        # DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
        p_diff = torch.where(p_diff < 0, 1.0, -1.0)

        grads = dtw_compute_full_per_channel(euc_d, p_diff, x.shape[1], w) # dims (n, k, d, i, i, j)

        return euc_d.sqrt(), grads

    dtw_compute_no_grad_per_channel(euc_d, w)

    return euc_d.sqrt(), None

# pylint: disable=invalid-name abstract-method arguments-differ
class torch_dtw_per_channel(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor, w: float):
        DTW, p_diff = dtw_fast_full_per_channel(x, y, w, compute_gradients=y.requires_grad)
        return DTW, p_diff

    @staticmethod
    def setup_context(ctx, _, output):
        _, p_diff = output
        ctx.save_for_backward(p_diff)

    @staticmethod
    def backward(ctx, dtw_grad, _):
        # dtw_grad dims (n, k, d, i, j) p_diff dims (n, k, d, i, i, j)
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:, :, :, :, None, :]) # dims (n, k, d, i, i, j)
        return None, mult.sum(dim=(-2, -1)).mean(dim=(0, 2)), None
