import torch
from networks_utils import SpatialAdaptation


class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, sep_mat, ycrop=0, xcrop=0, extension_factor=17, mode='bilinear'):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.sal = SpatialAdaptation(input_shape=grid_shape, T=True, R=True, Sc=False, Sh=False, mode=mode)
        self.bn = torch.nn.BatchNorm2d(1)
        self.lcrop, self.rcrop = xcrop, xcrop
        self.bcrop, self.tcrop = ycrop, ycrop

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
        # emg = self.bn(emg) # apply batch norm
        emg_sal = self.sal(emg).squeeze()
        emg_sal = emg_sal[:, self.tcrop:emg_sal.shape[1]-self.bcrop, self.lcrop:emg_sal.shape[2]-self.rcrop]
        extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
        sources = self.sep_mat(extended_emg)
        return sources

    def get_extended_emg(self, emg):
        '''Get the SAL, cropped + extended EMG from the original EMG.'''
        emg_sal = self.sal(emg).squeeze()
        emg_sal = emg_sal[:, self.tcrop:emg_sal.shape[1]-self.bcrop, self.lcrop:emg_sal.shape[2]-self.rcrop]
        extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
        return extended_emg
    
    def refine_sep_mat(self, extended_emg_sal, dts, inv_cov):
        '''Refine the separation vectors using the EMG and the estimated sources.'''
        new_sep_mat = torch.zeros(len(dts), extended_emg_sal.shape[0])
        for mu_idx in range(len(dts)):
            new_sep_mat[mu_idx, :] = extended_emg_sal[:, dts[mu_idx]].mean(dim=1)
        new_sep_mat = new_sep_mat @ inv_cov
        self.sep_mat.weight = torch.nn.Parameter(new_sep_mat.to(self.sep_mat.weight.device))


# class SpatialDecompositionAdaptation2(torch.nn.Module):    
#     # build the constructor
#     def __init__(self, grid_shape, sep_mat, ycrop=0, xcrop=0, extension_factor=17, mode='bilinear'):
#         super(SpatialDecompositionAdaptation, self).__init__()
#         self.grid_shape = grid_shape
#         self.nchans = torch.prod(torch.tensor(grid_shape))
#         self.sal = SpatialAdaptation(input_shape=grid_shape, T=True, R=True, Sc=False, Sh=False, mode=mode)
#         self.bn = torch.nn.BatchNorm2d(1)
#         self.lcrop, self.rcrop = xcrop, xcrop
#         self.bcrop, self.tcrop = ycrop, ycrop

#         self.sep_mat = torch.nn.Linear(sep_mat.shape[1], sep_mat.shape[0], bias=False)
#         with torch.no_grad():
#             self.sep_mat.weight.copy_(sep_mat)
        
#         self.extension_factor = extension_factor
    
#     def extend_emg(self, emg):
#         '''Extend the original EMG batch given extension factor.'''
#         device = emg.device
#         nchans = emg.shape[1]
#         extended_emg = torch.zeros((emg.shape[0] + self.extension_factor - 1, nchans*self.extension_factor)).to(device)
#         for idx in range(self.extension_factor):
#             extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
#         return extended_emg

#     # Extend, whiten and separate sources
#     def forward(self, emg):
#         # emg = self.bn(emg) # apply batch norm
#         emg_sal = self.sal(emg).squeeze()
#         emg_sal = emg_sal[:, self.tcrop:emg_sal.shape[1]-self.bcrop, self.lcrop:emg_sal.shape[2]-self.rcrop]
#         extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
#         sources = self.sep_mat(extended_emg)
#         return sources

#     def get_extended_emg(self, emg):
#         '''Get the SAL, cropped + extended EMG from the original EMG.'''
#         emg_sal = self.sal(emg).squeeze()
#         emg_sal = emg_sal[:, self.tcrop:emg_sal.shape[1]-self.bcrop, self.lcrop:emg_sal.shape[2]-self.rcrop]
#         extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
#         return extended_emg
    
#     def refine_sep_mat(self, extended_emg_sal, dts, inv_cov):
#         '''Refine the separation vectors using the EMG and the estimated sources.'''
#         new_sep_mat = torch.zeros(len(dts), extended_emg_sal.shape[0])
#         for mu_idx in range(len(dts)):
#             new_sep_mat[mu_idx, :] = extended_emg_sal[:, dts[mu_idx]].mean(dim=1)
#         new_sep_mat = new_sep_mat @ inv_cov
#         self.sep_mat.weight = torch.nn.Parameter(new_sep_mat.to(self.sep_mat.weight.device))
    

if __name__ == '__main__':
    sda = SpatialDecompositionAdaptation((20,20), sep_mat=torch.zeros(20,20,20))
    print(sda.sal.parameters())
