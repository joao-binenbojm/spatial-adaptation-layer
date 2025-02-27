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
from sal_decomposition.utils.grid_indexing import index_matrix4

if __name__ == '__main__':

    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    file = 'S1_25_Session1_MUEdit_edited.mat'
    file2 = 'S1_25_Session3_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    Tx_max, Ty_max, theta_max  = 2, 2, 10*np.pi/180
    torch.set_default_dtype(torch.float64)
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

    # Set pipeline parameters
    R = 16
    explained_var = 1-1e-3
    delta_width, delta_height = utils.out_of_bounds_pixels(H, W, theta_max)
    xcrop, ycrop = Tx_max + floor(delta_width + 0.5), Ty_max + floor(delta_height + 0.5)

    # Crop observations and get new sep_mat
    print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()

    # Get crop sep_mat
    if R == '1000/ch':
        R = 1000//(emg_grid_crop_train.shape[2]*emg_grid_crop_train.shape[3])
    extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
    # inv_cov = get_inv_cov_tikhonov(extended_emg_crop_train, reg=tikhonov)
    inv_cov = utils.get_inv_cov_torch(extended_emg_crop_train, explained_var=explained_var)
    sep_mat_crop_train = utils.get_sep_mat_torch(extended_emg_crop_train, mu_dts)
    sep_mat_crop_train = sep_mat_crop_train @ inv_cov

    # Initialize SDA module
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=sep_mat_crop_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
    sda.sal.mode = 'bicubic'

    with torch.no_grad():
        sources = sda(emg_grid)
    pred_dts, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)

    base_loss = utils.get_base_loss(emg_grid.to(torch.float64), sda, batch_size=batch_size, loss='kurtosis', device='cpu')
    loss_arr = utils.loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, T=None, bounds=(2.0, 2.0), batch_size=batch_size, num_points=20, loss='kurtosis', device=device)
    losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=100, lr=5e-4, boundaries=(2.0, 2.0, 10.0*np.pi/180), device='cuda')
    
    # Get new inverse covariance
    sda = sda.to('cpu')
    sda.sal.mode = 'bicubic'

    # Get minimum distance between original and transformed grid
    Tx, Ty, theta = -W*sda.sal.xshift.item()/2, -H*sda.sal.yshift.item()/2, -sda.sal.rot_theta.item()*np.pi
    original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
    print(f'MIN DISTANCE: {min_distance} pixels')

    # Create mask based on transformed coordinates being within convex 
    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

    # Obtain new separation matrix with all valid channels
    print('Obtaining new separation matrix...')
    sda.lcrop, sda.rcrop = lcrop, rcrop
    sda.bcrop, sda.tcrop = bcrop, tcrop
    emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop].clone()
    extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var)
    sep_mat_valid = utils.get_sep_mat_torch(extended_emg_valid, mu_dts)
    sep_mat_valid = sep_mat_valid @ inv_cov_valid

    # Update SDA module with new separation matrix
    sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
    
    # Get initial source estimates
    with torch.no_grad():
        sources = sda(emg_grid_test)
    pred_dts, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)


    # Get new covariance matrix
    print('Getting new inverse covariance...')
    sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=1-1e-12)
    sep_mat_test = utils.get_sep_mat_torch(extended_emg_test, pred_dts)
    sep_mat_test = sep_mat_test @ inv_cov_test
    with torch.no_grad():
        sources = (sep_mat_test @ extended_emg_test).T

    pred_dts, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)