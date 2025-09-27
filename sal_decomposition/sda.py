import torch
from networks_utils import SpatialAdaptation


### TEMPORARY UTILS TO BE INTEGRATED IN SDA LATER
def get_P(sal, sal_idx=0, align_corners=True, inverse=False):
    """
    Build dense bilinear interpolation matrix P for HxW grid given affine theta.
    P @ y_flat maps the EMG grid y into the transformed grid.
    """
    H, W = sal.input_shape
    M = H * W
    theta = sal.get_affine_transform(sal_idx=sal_idx, inverse=inverse)
    # Generate normalized affine sampling grid
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), size=(1, 1, H, W), align_corners=align_corners)[0]  # H x W x 2

    # Convert normalized coords to pixel indices
    if align_corners:
        x = ((grid[..., 0] + 1) * (W - 1) / 2).flatten() # align_corners=True
        y = ((grid[..., 1] + 1) * (H - 1) / 2).flatten()
    else:
        x = ((grid[..., 0] + 1) * W / 2 - 0.5).flatten() # align_corners=False
        y = ((grid[..., 1] + 1) * H / 2 - 0.5).flatten()

    # Neighbors
    # x0 = torch.floor(x).long().clamp(0, W - 1)
    # x1 = torch.ceil(x).long().clamp(0, W - 1)
    # y0 = torch.floor(y).long().clamp(0, H - 1)
    # y1 = torch.ceil(y).long().clamp(0, H - 1)
    x0 = torch.floor(x).long()
    x1 = torch.ceil(x).long()
    y0 = torch.floor(y).long()
    y1 = torch.ceil(y).long()

    # Weights
    dx = x - x0.float()
    dy = y - y0.float()

    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    def idx(y_idx, x_idx):
        return y_idx * W + x_idx

    i00 = idx(y0, x0)
    i01 = idx(y0, x1)
    i10 = idx(y1, x0)
    i11 = idx(y1, x1)

    # Mask for valid indices (inside grid) --> effectively treats out-of-bounds as zero-padding
    valid00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    valid01 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    valid10 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
    valid11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

    # Assemble dense P
    P = torch.zeros((M, M), dtype=torch.float32, device=theta.device)
    rows = torch.arange(M, device=theta.device)

    # P[rows, i00] += w00
    # P[rows, i01] += w01
    # P[rows, i10] += w10
    # P[rows, i11] += w11

    # Only add weights for valid indices
    P[rows[valid00], i00[valid00]] += w00[valid00]
    P[rows[valid01], i01[valid01]] += w01[valid01]
    P[rows[valid10], i10[valid10]] += w10[valid10]
    P[rows[valid11], i11[valid11]] += w11[valid11]

    return P

def apply_P_to_emg(emg, P):
    """
    Applies interpolation matrix P to EMG grid.
    Args:
        emg: Tensor of shape (B, 1, H, W)
        P: Interpolation matrix of shape (H*W, H*W)
    Returns:
        Transformed EMG of shape (B, 1, H, W)
    """
    B, _, H, W = emg.shape
    emg_flat = emg.view(B, H * W).T  # shape: (H*W, B)
    emg_trans = (P @ emg_flat).T     # shape: (B, H*W)
    return emg_trans.view(B, 1, H, W)

def reg_pinv(P, lam=1e-3):
    """
    Compute regularized pseudoinverse of P: P_inv = (P^T P + lam*I)^(-1) P^T
    """
    # P: (m x n) tensor
    Pt = P.transpose(-2, -1)
    n = P.shape[1]
    I = torch.eye(n, device=P.device, dtype=P.dtype)
    return torch.linalg.solve(Pt @ P + lam * I, Pt)

def get_P_inv_extended(sal, R, sal_idx=0, inverse=False):
    """
    Build block-diagonal extended inverse P for temporal extension.
    P: [M, M] dense
    R: number of temporal taps
    """
    P_inv = get_P(sal, sal_idx=sal_idx, inverse=inverse)  # [M, M]
    # P_inv = reg_pinv(P)  # dense for simplicity
    blocks = [P_inv for _ in range(R)]
    P_inv_ext = torch.block_diag(*blocks)  # [M*R, M*R]
    return P_inv_ext



class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, STA, inv_cov, ycrop=0, xcrop=0, extension_factor=17, mode='bilinear'):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.extension_factor = extension_factor
        self.sal = SpatialAdaptation(input_shape=grid_shape, T=True, R=True, Sc=True, Sh=False, mode=mode, circular=False, constrain_params=False)
        self.bn = torch.nn.BatchNorm2d(1)
        self.lcrop, self.rcrop = xcrop, xcrop
        self.bcrop, self.tcrop = ycrop, ycrop

        ## TESTING
        self.register_buffer("P_inv_extended", torch.eye(self.nchans*self.extension_factor), persistent=False) # initialize as identity
        # self.register_buffer("P_sep_mat", torch.randn_like(sep_mat), persistent=False) # initialize as identity
        # self.register_buffer("sep_mat", sep_mat) # initialize as identity
        # self.sep_mat = torch.nn.Linear(sep_mat.shape[1], sep_mat.shape[0], bias=False)
        # with torch.no_grad():
            # self.sep_mat.weight.copy_(sep_mat)
        
        self.register_buffer("STA", STA)
        self.register_buffer("inv_cov", inv_cov)


    def extend_emg(self, emg):
        '''Extend the original EMG batch given extension factor.'''
        device = emg.device
        nchans = emg.shape[1]
        extended_emg = torch.zeros((emg.shape[0] + self.extension_factor - 1, nchans*self.extension_factor)).to(device)
        for idx in range(self.extension_factor):
            extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
        return extended_emg[:-(self.extension_factor-1),:].T

    # Extend, whiten and separate sources
    def forward(self, emg, inverse=False):
        if self.training:
            self.P_inv_extended = get_P_inv_extended(self.sal, self.extension_factor, sal_idx=0, inverse=inverse) # if testing, assume we have P_inv_extended available to use
            self.P_sep_mat = self.STA @ self.P_inv_extended.T @ self.inv_cov
        extended_emg = self.extend_emg(emg.reshape(emg.shape[0], -1))
        sources = self.P_sep_mat @ extended_emg
        return sources.T

    def apply_affine(self, emg):
        '''Apply the current affine transformation to the EMG grid.'''
        P = get_P(self.sal, sal_idx=0, inverse=False)
        emg_transform = apply_P_to_emg(emg, P)
        return emg_transform

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


class SpatialDecompositionAdaptationOld(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, sep_mat, ycrop=0, xcrop=0, extension_factor=17, mode='bilinear'):
        super(SpatialDecompositionAdaptationOld, self).__init__()
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
    

if __name__ == '__main__':
    sda = SpatialDecompositionAdaptation((20,20), sep_mat=torch.zeros(20,20,20))
    print(sda.sal.parameters())
