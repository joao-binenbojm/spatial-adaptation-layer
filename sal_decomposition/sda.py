import torch
from networks_utils import SpatialAdaptation

class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, sep_mat, ycrop=0, xcrop=0, extension_factor=17):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.sal = SpatialAdaptation(input_shape=grid_shape, T=True, R=True, Sc=True, Sh=False)
        self.bn = torch.nn.BatchNorm2d(1)
        # self.ycrop = ycrop
        # self.xcrop = xcrop

        self.sep_mat = torch.nn.Linear(sep_mat.shape[1], sep_mat.shape[0], bias=False)
        with torch.no_grad():
            self.sep_mat.weight.copy_(sep_mat)
        
        self.extension_factor = extension_factor
    
    def extend_emg(self, emg):
        '''Extend the original EMG batch given extension factor.'''
        device = emg.device
        nchans = emg.shape[1]
        extended_emg = torch.zeros((emg.shape[0] + self.extension_factor - 1, nchans*self.extension_factor)).to(device)
        for idx in range(self.extension_factor):
            extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
        return extended_emg

    # Extend, whiten and separate sources
    def forward(self, emg):
        emg = self.bn(emg) # apply batch norm
        emg_sal = self.sal(emg).squeeze()
        extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
        sources = self.sep_mat(extended_emg)
        return sources