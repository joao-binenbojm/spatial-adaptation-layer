import torch
from networks_utils import SpatialAdaptation


# ### TEMPORARY UTILS TO BE INTEGRATED IN SDA LATER
# def get_P(sal, sal_idx=0, align_corners=True, inverse=False):
#     """
#     Build dense bilinear interpolation matrix P for HxW grid given affine theta.
#     P @ y_flat maps the EMG grid y into the transformed grid.
#     """
#     H, W = sal.input_shape
#     M = H * W
#     theta = sal.get_affine_transform(sal_idx=sal_idx, inverse=inverse)
#     # Generate normalized affine sampling grid
#     grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), size=(1, 1, H, W), align_corners=align_corners)[0]  # H x W x 2

#     # Convert normalized coords to pixel indices
#     if align_corners:
#         x = ((grid[..., 0] + 1) * (W - 1) / 2).flatten() # align_corners=True
#         y = ((grid[..., 1] + 1) * (H - 1) / 2).flatten()
#     else:
#         x = ((grid[..., 0] + 1) * W / 2 - 0.5).flatten() # align_corners=False
#         y = ((grid[..., 1] + 1) * H / 2 - 0.5).flatten()

#     # Neighbors
#     # x0 = torch.floor(x).long().clamp(0, W - 1)
#     # x1 = torch.ceil(x).long().clamp(0, W - 1)
#     # y0 = torch.floor(y).long().clamp(0, H - 1)
#     # y1 = torch.ceil(y).long().clamp(0, H - 1)
#     x0 = torch.floor(x).long()
#     x1 = torch.ceil(x).long()
#     y0 = torch.floor(y).long()
#     y1 = torch.ceil(y).long()

#     # Weights
#     dx = x - x0.float()
#     dy = y - y0.float()

#     w00 = (1 - dx) * (1 - dy)
#     w01 = dx * (1 - dy)
#     w10 = (1 - dx) * dy
#     w11 = dx * dy

#     def idx(y_idx, x_idx):
#         return y_idx * W + x_idx

#     i00 = idx(y0, x0)
#     i01 = idx(y0, x1)
#     i10 = idx(y1, x0)
#     i11 = idx(y1, x1)

#     # Mask for valid indices (inside grid) --> effectively treats out-of-bounds as zero-padding
#     valid00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
#     valid01 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
#     valid10 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
#     valid11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

#     # Assemble dense P
#     P = torch.zeros((M, M), dtype=torch.float32, device=theta.device)
#     rows = torch.arange(M, device=theta.device)

#     # P[rows, i00] += w00
#     # P[rows, i01] += w01
#     # P[rows, i10] += w10
#     # P[rows, i11] += w11

#     # Only add weights for valid indices
#     P[rows[valid00], i00[valid00]] += w00[valid00]
#     P[rows[valid01], i01[valid01]] += w01[valid01]
#     P[rows[valid10], i10[valid10]] += w10[valid10]
#     P[rows[valid11], i11[valid11]] += w11[valid11]

#     return P

def get_P(sal, sal_idx=0, input_shape=None, align_corners=True, inverse=False):
    """
    Build dense bilinear interpolation matrix P for HxW grid given affine theta.
    P @ y_flat maps the EMG grid y into the transformed grid.
    Out-of-boundary pixels are hard-zeroed (no mixing).
    """
    if input_shape is not None:
        H, W = input_shape
    else:
        H, W = sal.input_shape
        input_shape = sal.input_shape
    M = H * W
    theta = sal.get_affine_transform(sal_idx=sal_idx, input_shape=input_shape, inverse=inverse)
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), size=(1, 1, H, W), align_corners=align_corners)[0]  # H x W x 2

    if align_corners:
        x = ((grid[..., 0] + 1) * (W - 1) / 2).flatten()
        y = ((grid[..., 1] + 1) * (H - 1) / 2).flatten()
    else:
        x = ((grid[..., 0] + 1) * W / 2 - 0.5).flatten()
        y = ((grid[..., 1] + 1) * H / 2 - 0.5).flatten()

    x0 = torch.floor(x).long()
    x1 = torch.ceil(x).long()
    y0 = torch.floor(y).long()
    y1 = torch.ceil(y).long()

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

    # Validity masks for all four neighbors
    valid00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    valid01 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    valid10 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
    valid11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

    # Only interpolate if all neighbors are valid
    all_valid = valid00 & valid01 & valid10 & valid11

    P = torch.zeros((M, M), dtype=torch.float32, device=theta.device)
    rows = torch.arange(M, device=theta.device)

    # Only add weights for fully valid pixels
    P[rows[all_valid], i00[all_valid]] += w00[all_valid]
    P[rows[all_valid], i01[all_valid]] += w01[all_valid]
    P[rows[all_valid], i10[all_valid]] += w10[all_valid]
    P[rows[all_valid], i11[all_valid]] += w11[all_valid]

    # All other rows remain zero (hard-zeroed)

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

def block_extension(matrix, extension_factor):
    blocks = [matrix for _ in range(extension_factor)]
    extended_matrix = torch.block_diag(*blocks)
    return extended_matrix

def get_P_inv_extended(sal, R, sal_idx=0, inverse=False):
    """
    Build block-diagonal extended inverse P for temporal extension.
    P: [M, M] dense
    R: number of temporal taps
    """
    P_inv = get_P(sal, sal_idx=sal_idx, inverse=inverse)  # [M, M]
    P_inv_ext = block_extension(P_inv, extension_factor=R)
    return P_inv_ext


class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, STA, inv_cov, ycrop=0, xcrop=0, extension_factor=17, mode='bilinear'):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.extension_factor = extension_factor
        self.sal = SpatialAdaptation(input_shape=grid_shape, T=True, R=True, Sc=False, Sh=False, mode=mode, circular=False, constrain_params=False)
        self.bn = torch.nn.BatchNorm2d(1)
        self.xcrop, self.ycrop = xcrop, ycrop
        self.lcrop, self.rcrop = xcrop, xcrop
        self.bcrop, self.tcrop = ycrop, ycrop

        ## TESTING
        self.register_buffer("P_inv_extended", torch.eye(self.nchans*self.extension_factor), persistent=False) # initialize as identity
        self.register_buffer("P_sep_mat", torch.randn_like(STA), persistent=False) # initialize as identity
        self.channel_scales = torch.nn.Parameter(torch.ones(self.nchans))
        self.register_buffer("crop_mask", self.get_crop_mask()) # initialize crop mask to be applied to extended_emg        
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
        # if self.training:
        # scalings = torch.diag(self.channel_scales)
        # D_inv_extended = block_extension(scalings, extension_factor=self.extension_factor)
        # self.update_cropping()
        # self.P_inv_extended = get_P_inv_extended(self.sal, self.extension_factor, sal_idx=0, inverse=inverse) # if testing, assume we have P_inv_extended available to use
        # self.STA_transform = (self.STA @ self.P_inv_extended.T)[:, self.crop_mask] # 
        # self.P_sep_mat = self.STA_transform @ self.inv_cov

        # sep_mat = self.STA @ self.inv_cov 
        # self.P_sep_mat = self.spatial_transform_filter(sep_mat) #[:, self.crop_mask]

        # self.P_sep_mat = self.P_sep_mat @ D_inv_extended
        self.P_sep_mat = self.spatial_transform_filter(self.STA)[:, self.crop_mask] @ self.inv_cov
        
        # Transform mask spatially
        # STA_transform = self.spatial_transform_filter(self.STA)

        # # Apply hard mask to transformed STA
        # extended_mask_flat = self.get_mask()  # Shape: (M*R,)
        # STA_transform_masked = STA_transform * extended_mask_flat.unsqueeze(0)  # [N, M*R]
        # inv_cov_masked = self.inv_cov * extended_mask_flat.unsqueeze(1) * extended_mask_flat.unsqueeze(0)
        
        # Get masked separation matrix
        # self.P_sep_mat = STA_transform_masked @ inv_cov_masked
        
        # self.P_sep_mat = STA_transform @ self.inv_cov
        # self.P_sep_mat = self.STA @ self.inv_cov

        emg = emg[:, :, self.tcrop:emg.shape[2]-self.bcrop, self.lcrop:emg.shape[3]-self.rcrop] # apply cropping
        extended_emg = self.extend_emg(emg.reshape(emg.shape[0], -1))
        sources = self.P_sep_mat @ extended_emg
        return sources.T

    # def update_cropping(self):
    #     """Every forward pass, assume transformation has changed and compute the new valid crop."""
    #     H, W = self.grid_shape
    #     Tx, Ty = (W-1)*self.sal.xshift[0]/2, (H-1)*self.sal.yshift[0]/2
    #     self.lcrop = torch.ceil(Tx).to(torch.int) if Tx > 0 else torch.tensor([0])
    #     self.rcrop = self.xcrop - self.lcrop
    #     self.tcrop = torch.ceil(Ty).to(torch.int) if Ty > 0 else torch.tensor([0])
    #     self.bcrop = self.ycrop - self.tcrop
    #     self.crop_mask = self.get_crop_mask(grid_shape=self.grid_shape)
    
    # def get_mask(self):
    #     ''' Get the spatial mask based on current transformation parameters.'''
    #     H, W = self.grid_shape
    #     theta = self.sal.get_affine_transform(input_shape=(H, W)).to(self.P_sep_mat.device)
    #     theta = theta.repeat(1, 1, 1)
    #     grid = self.sal.get_grid(theta, input_shape=(H, W))
    #     in_bounds_mask = ((grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)).float().detach()
    #     spatial_mask_flat = in_bounds_mask.reshape(-1)  # Shape: (M,)
    #     extended_mask_flat = spatial_mask_flat.repeat(self.extension_factor)  # Shape: (M*R,)
    #     return extended_mask_flat

    def spatial_transform_filter(
        self,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies a spatial transformation to a convolutive separation matrix
        with the memory layout [M, (C_lag0, C_lag1, ...)].

        Args:
            B: The separation matrix of shape [M, (H*W)*L].
            theta: The 2x3 affine transformation matrix.
            H, W, L: The height, width, and extension factor (R).

        Returns:
            The spatially transformed separation matrix of shape [M, (H*W)*L].
        """
        H, W = self.grid_shape
        M = B.shape[0]
        C = H*W

        # == 1. UNFOLD ==
        # Start with shape [M, C*L], where data is grouped by lag.
        # Reshape to isolate the time lag (L) and channel (C) dimensions.
        B_unfolded = B.reshape(M, self.extension_factor, C)

        # Further reshape to isolate the spatial H and W dimensions.
        # The shape is now [M, self.extension_factor, H, W].
        B_spatial = B_unfolded.reshape(M, self.extension_factor, H, W)

        # Reshape for batch processing. We combine M and L into the batch dimension.
        # The new shape is [M*self.extension_factor, 1, H, W] for grid_sample.
        B_reshaped_for_grid_sample = B_spatial.reshape(M * self.extension_factor, 1, H, W)


        # == 2. APPLY TRANSFORMATION ==
        # Create the sampself.extension_factoring grid.
        B_transformed = self.sal(B_reshaped_for_grid_sample)

        # == 3. REFOLD ==
        # Reshape back to separate the M and L dimensions: [M, self.extension_factor H, W].
        B_refolded_spatial = B_transformed.reshape(M, self.extension_factor, H, W)

        # Flatten back to the final desired shape [M, L*(H*W)] = [M, C*L].
        B_final = B_refolded_spatial.reshape(M, self.extension_factor * C)

        return B_final
    
    def get_crop_mask(self):
        ''' Apply the equivalent cropping operation by transforming an equivalent binary mask the same way as the EMG image frames are transformed.'''
        mask = torch.zeros(self.grid_shape[0], self.grid_shape[1])
        mask[self.tcrop:mask.shape[0]-self.bcrop, self.lcrop:mask.shape[1]-self.rcrop] = 1 # channels to keep
        mask = mask.reshape(-1)
        extended_crop_mask = mask.repeat(self.extension_factor)
        return extended_crop_mask.squeeze().to(torch.bool)

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
    def forward(self, emg, inverse=False):
        # emg = self.bn(emg) # apply batch norm
        emg_sal = self.sal(emg, inverse=inverse).squeeze()
        emg_sal = emg_sal[:, self.tcrop:emg_sal.shape[1]-self.bcrop, self.lcrop:emg_sal.shape[2]-self.rcrop]
        extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))[:-(self.extension_factor-1)]
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
