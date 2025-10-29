import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from math import floor

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2


import torch
import torch.fft

# def normalized_cross_correlation_3d(A: torch.Tensor, B: torch.Tensor, 
#                                     min_overlap_ratio: float = 0.3) -> torch.Tensor:
#     """
#     Compute true normalized cross-correlation (NCC) between two 3D tensors.
    
#     Args:
#         A: 3D tensor, shape (D, H, W)
#         B: 3D tensor, shape (D, H, W)
#         min_overlap_ratio: Minimum fraction of B's volume that must overlap
#                           with A for valid correlation (default: 0.3 = 30%)
    
#     Returns:
#         NCC: 3D tensor of shape (D_out, H_out, W_out)
#              where D_out = D_A + D_B - 1, etc.
#              Valid entries are between -1 and 1, representing the correlation
#              coefficient at each lag position. Invalid regions (insufficient
#              overlap) are set to -inf or a large negative value.
#     """
    
#     # Ensure float
#     A = A.float()
#     B = B.float()
    
#     # Zero-mean
#     A_mean = A.mean()
#     B_mean = B.mean()
#     A_zm = A - A_mean
#     B_zm = B - B_mean
    
#     # Output shape for linear correlation
#     out_shape = [A.shape[i] + B.shape[i] - 1 for i in range(3)]
    
#     # 1. Compute numerator: cross-correlation of zero-meaned signals
#     FA = torch.fft.fftn(A_zm, out_shape)
#     FB = torch.fft.fftn(B_zm, out_shape)
#     cross_corr = torch.fft.ifftn(FA * torch.conj(FB)).real
    
#     # 2. Compute denominator: local energy normalization
#     # We need sqrt(sum(A_overlap²) * sum(B_overlap²)) at each lag
    
#     # Also compute overlap count to mask edge effects
#     ones_B = torch.ones_like(B)
#     F_ones_B = torch.fft.fftn(ones_B, out_shape)
    
#     # Compute local sum of A² using convolution with ones
#     A_sq = A_zm ** 2
    
#     # Sum of A² over regions matching B's shape (sliding window)
#     FA_sq = torch.fft.fftn(A_sq, out_shape)
#     local_sum_A_sq = torch.fft.ifftn(FA_sq * torch.conj(F_ones_B)).real
    
#     # Count of overlapping voxels at each position
#     FA_ones = torch.fft.fftn(torch.ones_like(A), out_shape)
#     overlap_count = torch.fft.ifftn(FA_ones * torch.conj(F_ones_B)).real
    
#     # Compute local sum of B² using convolution with ones
#     B_sq = B_zm ** 2
#     ones_A = torch.ones_like(A)
    
#     # Sum of B² over regions matching A's shape (sliding window)
#     FB_sq = torch.fft.fftn(B_sq, out_shape)
#     F_ones_A = torch.fft.fftn(ones_A, out_shape)
#     local_sum_B_sq = torch.fft.ifftn(F_ones_A * torch.conj(FB_sq)).real
    
#     # 3. Compute minimum required overlap
#     B_volume = B.shape[0] * B.shape[1] * B.shape[2]
#     min_overlap_voxels = min_overlap_ratio * B_volume
    
#     # 3. Compute normalization factor at each position
#     # norm[i,j,k] = sqrt(local_sum_A_sq[i,j,k] * local_sum_B_sq[i,j,k])
#     local_norm = torch.sqrt(local_sum_A_sq * local_sum_B_sq)
    
#     # 4. Handle numerical issues
#     # Clamp negative values that might arise from numerical errors
#     local_sum_A_sq = torch.clamp(local_sum_A_sq, min=0.0)
#     local_sum_B_sq = torch.clamp(local_sum_B_sq, min=0.0)
    
#     # Compute normalization factor at each position
#     local_norm = torch.sqrt(local_sum_A_sq * local_sum_B_sq)
    
#     # 5. Create validity mask
#     # Valid only if: (1) sufficient overlap AND (2) non-zero local energy
#     epsilon = 1e-10
#     valid_mask = (overlap_count >= min_overlap_voxels) & (local_norm > epsilon)
    
#     # 6. Compute NCC only where valid
#     # Initialize with very negative value for invalid regions
#     NCC = torch.full_like(cross_corr, -float('inf'))
    
#     # Compute NCC only where valid
#     NCC[valid_mask] = cross_corr[valid_mask] / local_norm[valid_mask]
    
#     # Clamp valid values to [-1, 1] to handle any remaining numerical errors
#     NCC[valid_mask] = torch.clamp(NCC[valid_mask], -1.0, 1.0)
    
#     # Replace any remaining NaN or Inf in valid region with 0
#     NCC = torch.where(valid_mask, torch.nan_to_num(NCC, nan=0.0, posinf=0.0, neginf=0.0), NCC)
    
#     return NCC

def sum_of_squared_differences_3d(A: torch.Tensor, B: torch.Tensor,
                                   min_overlap_ratio: float = 0.3,
                                   normalize: bool = True) -> torch.Tensor:
    """
    Sum of Squared Differences (SSD) - simpler, less biased.
    Lower values = better match (unlike NCC where higher is better)
    
    Args:
        normalize: If True, divide by overlap size for fair comparison
    
    Returns: SSD at each position (lower is better, set to +inf for invalid)
    """
    A = A.float()
    B = B.float()
    
    out_shape = [A.shape[i] + B.shape[i] - 1 for i in range(3)]
    
    # Compute sum(A²) locally
    ones_B = torch.ones_like(B)
    F_ones_B = torch.fft.fftn(ones_B, out_shape)
    
    A_sq = A ** 2
    FA_sq = torch.fft.fftn(A_sq, out_shape)
    sum_A_sq = torch.fft.ifftn(FA_sq * torch.conj(F_ones_B)).real
    
    # Compute sum(B²) locally
    B_sq = B ** 2
    ones_A = torch.ones_like(A)
    FB_sq = torch.fft.fftn(B_sq, out_shape)
    F_ones_A = torch.fft.fftn(ones_A, out_shape)
    sum_B_sq = torch.fft.ifftn(F_ones_A * torch.conj(FB_sq)).real
    
    # Compute cross-correlation (for -2*sum(A*B) term)
    FA = torch.fft.fftn(A, out_shape)
    FB = torch.fft.fftn(B, out_shape)
    cross_corr = torch.fft.ifftn(FA * torch.conj(FB)).real
    
    # Compute overlap count (needed for both normalization and masking)
    FA_ones = torch.fft.fftn(torch.ones_like(A), out_shape)
    overlap_count = torch.fft.ifftn(FA_ones * torch.conj(F_ones_B)).real
    overlap_count = torch.clamp(overlap_count, min=0.0)
    
    # SSD = sum((A-B)²) = sum(A²) + sum(B²) - 2*sum(A*B)
    ssd = sum_A_sq + sum_B_sq - 2 * cross_corr
    
    # Clamp to handle numerical errors (SSD should never be negative)
    ssd = torch.clamp(ssd, min=0.0)
    
    # Normalize by overlap size if requested
    # This gives you mean squared error per voxel instead of total error
    if normalize:
        ssd = ssd / overlap_count
    
    # Mask invalid regions (insufficient overlap)
    valid_mask = torch.ones(out_shape, dtype=torch.bool, device=A.device)
    
    for dim in range(3):
        # Calculate overlap range in this dimension
        # B spans from index i to i+B.shape[dim]
        # A spans from 0 to A.shape[dim]
        # Overlap exists when these ranges intersect
        
        # For each position along this dimension, compute overlap length
        overlap_dim = torch.zeros(out_shape[dim], device=A.device)
        for pos in range(out_shape[dim]):
            B_start = pos
            B_end = pos + B.shape[dim]
            A_start = 0
            A_end = A.shape[dim]
            
            # Intersection of [B_start, B_end) and [A_start, A_end)
            overlap_start = max(B_start, A_start)
            overlap_end = min(B_end, A_end)
            overlap_length = max(0, overlap_end - overlap_start)
            overlap_dim[pos] = overlap_length
        
        # Require minimum overlap in this dimension
        min_overlap_dim = min_overlap_ratio * B.shape[dim]
        dim_valid = overlap_dim >= min_overlap_dim
        
        # Broadcast to full shape and combine with mask
        if dim == 0:
            valid_mask &= dim_valid.view(-1, 1, 1).expand(out_shape)
        elif dim == 1:
            valid_mask &= dim_valid.view(1, -1, 1).expand(out_shape)
        else:
            valid_mask &= dim_valid.view(1, 1, -1).expand(out_shape)
    
    ssd[~valid_mask] = float('inf')

    return ssd


if __name__ == '__main__':

    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s2_edited'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    file = 'S2_25_Session1_MUEdit_edited.mat'
    file2 = 'S2_25_Session3_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 2048
    bounds = [3, 3] #, 10*np.pi/180]
    # torch.set_default_dtype(torch.float64)
    print(os.listdir(DIR))

    # Load training data
    signal, edition = utils.open_mat_output(DIR, file)
    start, end = utils.get_target_boundaries(signal['target'].squeeze())
    print('FILTERING TRAINING DATA...')
    emg = signal['data'][:, start:end]
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)
    emg_grid = utils.make_grid(emg, index_matrix4)
    H, W = emg_grid.shape[2], emg_grid.shape[3]
    Nch = H*W
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    emg_grid = utils.handle_outliers(emg_grid)
    emg_grid = emg_grid / (emg_grid.std() + 1e-12)
    # Load discharge times
    dts = edition['Dischargetimes']
    mu_dts = utils.squeeze_dts(dts)
    mu_dts = utils.filter_dts(mu_dts, start, end)

    # Load test data
    signal2, edition2 = utils.open_mat_output(DIR, file2)
    start2, end2 = utils.get_target_boundaries(signal2['target'].squeeze())
    emg2 = signal2['data'][:, start2:end2]
    emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12) # centering emg
    print('FILTERING TEST DATA...')
    emg2 = bandpass_filter(notch_filter(emg2, fsamp=fsamp), fsamp=fsamp)
    emg_grid_test = utils.make_grid(emg2, index_matrix4)
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    emg_grid_test = utils.handle_outliers(emg_grid_test)
    emg_grid_test = emg_grid_test / (emg_grid_test.std() + 1e-12)
    # Load discharge times
    dts2 = edition2['Dischargetimes']
    mu_dts2 = utils.squeeze_dts(dts2)
    mu_dts2 = utils.filter_dts(mu_dts2, start2, end2)

    # Get top MUAP from session 1, and compare with MUAPs from session 2
    muap = utils.get_sta_muaps(emg_grid, mu_dts[0].astype(int), L=150, spacing=1.2, plot=True)

    min_ssds = []
    min_locations = []
    for idx in range(len(mu_dts2)):
        muaps1 = utils.get_sta_muaps(emg_grid_test, mu_dts2[idx].astype(int), L=150, spacing=1.2, plot=False)
        ssd = sum_of_squared_differences_3d(muap, muaps1)
        flat_idx = torch.argmin(ssd)
        locations = torch.unravel_index(flat_idx, ssd.shape)
        locations = [loc - (muap.shape[l_idx] - 1) for l_idx, loc in enumerate(locations)]
        print(f'MU #{idx+1}: SSD = {ssd.min()}, location: {locations}')
        min_ssds.append(ssd.min())
        min_locations.append(locations)


    print(f'BEST MUAP: {np.argmin(min_ssds)}, LOCATION: {min_locations[np.argmin(min_ssds)]}, SSD: {np.min(min_ssds)}')
    print()

    # # Set pipeline parameters
    # R = 16
    # explained_var = 1-1e-3
    # delta_width, delta_height = utils.out_of_bounds_pixels(H, W, 0.0)
    # xcrop, ycrop = bounds[0] + floor(delta_width + 0.5), bounds[1] + floor(delta_height + 0.5)

    # # Crop observations and get new sep_mat
    # # print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    # # emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()

    # # Get crop sep_mat
    # extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
    # inv_cov_train = utils.get_inv_cov_torch(extended_emg_train, explained_var=explained_var).to(torch.float32)
    # STA = utils.get_sta_templates(extended_emg_train, mu_dts).to(torch.float32)

    # # Initialize SDA module
    # sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, extension_factor=R)
    # # sda.sal.mode = 'bicubic'

    # with torch.no_grad():
    #     sources = sda(emg_grid)
    # pred_dts_train, sils = utils.get_silohuette(sources)
    # matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts_train, fs=fsamp)
    # print('F1-Score Training:', np.mean(f1_scores))

    # # Get base loss so we can understand how much sparsity relative to the training set/original decomposition
    # base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss='kurtosis', device='cpu')
    
    # # Get inverse covariance of the test grid, and determine the spatial transformation required for the STA templates to become optimal separation vectors
    # sda.lcrop, sda.rcrop = xcrop, xcrop
    # sda.tcrop, sda.bcrop = ycrop, ycrop
    # sda.crop_mask = sda.get_crop_mask()
    # emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
    # extended_emg_test = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
    # inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=explained_var).to(torch.float32)
    # sda.inv_cov = inv_cov_test

    # sources, losses = utils.search_fit_sda(emg_grid_test.clone(), sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=1000, nepochs=100, lr=5e-4, boundaries=bounds, device='cuda')
    
    # # Get new inverse covariance
    # sda = sda.to('cpu')
    # # sda.sal.mode = 'bicubic'

    # # Get minimum distance between original and transformed grid
    # Tx, Ty, theta = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2, sda.sal.rot_theta[0].item()*np.pi
    # original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
    # print(f'MIN DISTANCE: {min_distance} pixels')

    # # Create mask based on transformed coordinates being within convex 
    # lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

    # # Obtain new separation matrix with all valid channels
    # # print('Obtaining new separation matrix...')
    # # emg_grid_test_crop = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop].clone()
    # # extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # # inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var)
    # # sep_mat_valid = utils.get_sep_mat_torch(extended_emg_valid, mu_dts)
    # # sep_mat_valid = sep_mat_valid @ inv_cov_valid

    # # Update SDA module with new separation matrix
    # # sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
    # # print('Obtaining new separation matrix...')
    # # sda.lcrop, sda.rcrop = lcrop, rcrop
    # # sda.bcrop, sda.tcrop = bcrop, tcrop
    # # emg_grid_valid = emg_grid_test[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
    # # extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # # inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var).to(torch.float32)
    # # sda.inv_cov = inv_cov_valid
    # # sda.crop_mask = sda.get_crop_mask()

    # # Get initial source estimates
    # with torch.no_grad():
    #     sources = sda(emg_grid_test)

    # # Get performance on new test grid post training
    # pred_dts, sils = utils.get_silohuette(sources)
    # matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts2, pred_dts, fs=fsamp)
    # print(f1_scores)

    # # # Get new covariance matrix
    # # print('Getting new inverse covariance...')
    # # sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
    # # extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    # # inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=1-1e-12)
    # # sep_mat_test = utils.get_sep_mat_torch(extended_emg_test, pred_dts)
    # # sep_mat_test = sep_mat_test @ inv_cov_test
    # # with torch.no_grad():
    # #     sources = (sep_mat_test @ extended_emg_test).T

    # # pred_dts, sils = utils.get_silohuette(sources)
    # # matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    # # print(f1_scores)