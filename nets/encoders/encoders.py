from torch import nn

class NoLayer(nn.Module):
    def __init__(self, channels, ref_size, 
            wdw_size, n_feature_maps) -> None:
        super().__init__()

        self.channels = channels
        self.ref_size = ref_size
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

    def get_output_shape(self):
        return (-1, self.channels, self.ref_size, self.wdw_size)
    
    def forward(self, x):
        return x
    
class RemoveUpperPart(nn.Module):
    def __init__(self, channels, ref_size, 
            wdw_size, n_feature_maps) -> None:
        super().__init__()

        self.channels = channels
        self.ref_size = ref_size
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

    def get_output_shape(self):
        return (-1, self.channels, self.wdw_size)
    
    def forward(self, x):
        return x[:,:,-1,:]