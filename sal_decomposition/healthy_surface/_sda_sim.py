import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from math import floor
from scipy.interpolate import griddata

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4

if __name__ == '__main__':

    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/2mm'
    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s2_edited'
    print(os.listdir(DIR))
    file = 'S2_25_Session1_MUEdit_edited.mat'
    # file = 'S1_25_2mm_Session1_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    Tx_max, Ty_max, theta_max  = 2, 2, 10*np.pi/180
    signal, edition = utils.open_mat_output(DIR, file)
    start, end = utils.get_target_boundaries(signal['target'].squeeze())
    torch.set_default_dtype(torch.float64)

    # Apply filters to data and reshape into desired shape
    print('FILTER DATA...')
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

    # Get separation matrix
    dts = edition['Dischargetimes']
    mu_dts = utils.squeeze_dts(dts)
    mu_dts = utils.filter_dts(mu_dts, start, end)

    R = '1000/ch'
    explained_var = 1-1e-3
    Tx, Ty, theta = -1.25, 0.75, 0.0 #10*np.pi/180 # test integer shift
    delta_width, delta_height = utils.out_of_bounds_pixels(H, W, theta_max)
    xcrop, ycrop = Tx_max + floor(delta_width + 0.5), Ty_max + floor(delta_height + 0.5)

    # Crop observations and get new sep_mat
    print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
    emg_grid_test = utils.apply_affine(emg_grid, Tx, Ty, theta, mode='bicubic') #theta) # Apply affine transformation and appropriate padding
    # emg_grid_test = apply_affine(emg_grid_test, -Tx, -Ty, 0, mode='bicubic')

    # Get minimum distance between original and transformed grid
    original_grid = utils.get_transformed_grid((1, 1, H, W))
    transformed_grid = utils.get_transformed_grid((1, 1, H, W), Tx, Ty, theta)
    center = transformed_grid[0:1, H//2:H//2 + 1, W//2:W//2 + 1, :]
    distances = torch.linalg.norm(original_grid - center, dim=3)
    min_distance = torch.min(distances)
    print(f'MIN DISTANCE: {min_distance} pixels')

    # Create mask based on transformed coordinates being within convex 
    original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
    print(f'MIN DISTANCE: {min_distance} pixels')

    # Create mask based on transformed coordinates being within convex 
    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

    # Test that masking channels is a valid solution
    print('TESTING MASKING CHANNELS...')
    emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
    if R == '1000/ch':
        R = 1000//(emg_grid_valid.shape[2]*emg_grid_valid.shape[3])

    extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # inv_cov_valid = get_inv_cov_tikhonov(extended_emg_valid, reg=tikhonov)
    inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var)
    # inv_cov_valid = get_inv_cov_torch(extended_emg_valid, explained_var=1.0-1e-4)
    sep_mat_valid = utils.get_sep_mat_torch(extended_emg_valid, mu_dts)
    sep_mat_valid = sep_mat_valid @ inv_cov_valid
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=sep_mat_valid, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
    sda.lcrop, sda.rcrop = lcrop, rcrop
    sda.bcrop, sda.tcrop = bcrop, tcrop

    # Add optimal parameters for testing
    sda.sal.mode = 'bicubic'
    with torch.no_grad():
        sda.sal.xshift.copy_(-2*Tx/W)
        sda.sal.yshift.copy_(-2*Ty/H)
        sda.sal.rot_theta.copy_(-theta/np.pi)

    with torch.no_grad():
        source_est_valid = sda(emg_grid_test)
    pred_dts, sils = utils.get_silohuette(source_est_valid)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(np.mean(f1_scores))

    # lcrop, rcrop, bcrop, tcrop = lcrop + 1, rcrop + 1, bcrop + 1, tcrop + 1

    # Get crop sep_mat
    if R == '1000/ch':
        R = 1000//(emg_grid_crop_train.shape[2]*emg_grid_crop_train.shape[3])
    extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
    # inv_cov = get_inv_cov_tikhonov(extended_emg_crop_train, reg=tikhonov)
    inv_cov = utils.get_inv_cov_torch(extended_emg_crop_train, explained_var=explained_var)
    sep_mat_crop_train = utils.get_sep_mat_torch(extended_emg_crop_train, mu_dts)
    sep_mat_crop_train = sep_mat_crop_train @ inv_cov
    sda.sep_mat.weight = torch.nn.Parameter(sep_mat_crop_train)
    sda.lcrop, sda.rcrop = xcrop, xcrop
    sda.bcrop, sda.tcrop = ycrop, ycrop

    # Reset SDA-SAL parameters
    with torch.no_grad():
        sda.sal.xshift.copy_(0.0)
        sda.sal.yshift.copy_(0.0)
        sda.sal.rot_theta.copy_(0.0)

    # Test that SDA is working
    with torch.no_grad():
        source_est_crop = sda(emg_grid)
    
    pred_dts, sils = utils.get_silohuette(source_est_crop)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(np.mean(f1_scores))

    # # Reset SDA-SAL parameters
    # with torch.no_grad():
    #     sda.sal.xshift.copy_(0.0)
    #     sda.sal.yshift.copy_(0.0)
    #     sda.sal.rot_theta.copy_(0.0)

    # Filter based on f1_score
    # mu_dts_train = [f1_score > 0.7 for f1_score in f1_scores]
    # sda.sep_mat.weight = torch.nn.Parameter(sep_mat_crop_train[mu_dts_train, :])

    base_loss = utils.get_base_loss(emg_grid.to(torch.float64), sda, batch_size=batch_size, loss='kurtosis', device='cpu')

    sda.sal.mode = 'bicubic' # 'bicubic'
    ## Test that separation matrix is working
    # source_est_crop = sep_mat_crop_train @ extended_emg_crop_train
    # spktrain = torch.zeros(source_est_crop.shape[0])
    # spktrain[mu_dts[1]] = 1.0
    # plt.figure()
    # plt.plot(source_est_crop[0,:1000]/source_est_crop[0,:1000].max())
    # plt.plot(spktrain[:1000])
    # # plt.plot(edition['Pulsetrain'][0,0][0,start:start+1000] / edition['Pulsetrain'][0,0][0,start:start+1000].max())
    # plt.legend(['Source Estimate', 'Pulse Trains'])
    # plt.savefig('test_sep_mat')

    sda.sal.mode = 'bilinear'
    loss_arr = utils.loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, T=(Tx, Ty), bounds=(2.0, 2.0), batch_size=batch_size, num_points=20, loss='kurtosis', device=device)
    losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=50, lr=5e-3, boundaries=(2.5, 2.5, 10.0*np.pi/180), device='cuda')
    
    # Get new inverse covariance
    sda = sda.to('cpu')
    sda.sal.mode = 'bicubic'

    # Obtain new separation matrix with all valid channels
    print('Obtaining new separation matrix...')
    sda.lcrop, sda.rcrop = lcrop, rcrop
    sda.bcrop, sda.tcrop = bcrop, tcrop
    emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
    if R == '1000/ch':
        R = 1000/(emg_grid_valid.shape[2]*emg_grid_valid.shape[3])
    # emg_grid_valid = emg_grid * mask # mask out channels that are out of bounds
    extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # inv_cov_valid = get_inv_cov_tikhonov(extended_emg_valid, reg=tikhonov)
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
    if R == '1000/ch':
        R = 1000/(emg_grid_test.shape[2]*emg_grid_test.shape[3])
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=1-1e-12)
    sep_mat_test = utils.get_sep_mat_torch(extended_emg_test, pred_dts)
    sep_mat_test = sep_mat_test @ inv_cov_test
    with torch.no_grad():
        sources = (sep_mat_test @ extended_emg_test).T
    # pred_dts, sils = get_silohuette(sources)
    # # Refine separation matrix
    # N = 50
    # print('REFINING SEPARATION MATRIX...')
    # prev_cv, cv, idx = -np.inf, -1e6, 0
    # with torch.no_grad():
    #     # for idx in tqdm(range(N)):
    #     while cv > prev_cv:
    #         prev_cv = cv
    #         sda.refine_sep_mat(extended_emg_sal, pred_dts, inv_cov_test)
    #         sources = sda(emg_grid_test)
    #         pred_dts, sils = get_silohuette(sources)
    #         cv = average_cv(pred_dts)

    #         print(f'Update #{idx}')
    #         print(cv)
    #         idx += 1

    pred_dts, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)

  
        