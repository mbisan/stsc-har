import torch

from utils.dtw.dtw import torch_dtw
from utils.dtw.dtw_per_channel import torch_dtw_per_channel
from utils.dtw.dtw_pycuda import torch_dtw_cuda

class DTWLayer(torch.nn.Module):
    def __init__(self, n_patts, d_patts, l_patts, l_out: int = None, rho: float = 1) -> None:
        super().__init__()

        if l_out is None:
            self.l_out = l_patts
        else:
            self.l_out = l_out

        self.w: torch.float32 = rho ** (1/l_patts)
        self.patts = torch.nn.Parameter(torch.randn(n_patts, d_patts, l_patts))
    
    def forward(self, x):
        return torch_dtw.apply(x, self.patts, self.w)[0][:,:,:,-self.l_out:]
    
class DTWLayerPerChannel(torch.nn.Module):
    def __init__(self, n_patts, d_patts, l_patts, l_out: int = None, rho: float = 1) -> None:
        super().__init__()

        if l_out is None:
            self.l_out = l_patts
        else:
            self.l_out = l_out

        self.w: torch.float32 = rho ** (1/l_patts)
        self.patts = torch.nn.Parameter(torch.randn(n_patts, l_patts))
    
    def forward(self, x):
        y = torch_dtw_per_channel.apply(x, self.patts, self.w)[0][:,:,:,:,-self.l_out:]
        return y.reshape((y.shape[0], y.shape[1]*y.shape[2], y.shape[3], y.shape[4]))
    
class DTWFeatures(torch.nn.Module):
    def __init__(self, n_patts, d_patts, l_patts, l_out: int = 0, rho: float = 1) -> None:
        super().__init__()

        self.w: torch.float32 = rho ** (1/l_patts)
        self.patts = torch.nn.Parameter(torch.randn(n_patts, d_patts, l_patts))
    
    def forward(self, x):
        x = torch_dtw_cuda.apply(x, self.patts, self.w)
        return x.sqrt()