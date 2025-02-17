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
from sal_decomposition.utils import *
from loss_functions import KurtosisLoss, NegentropyLoss
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay


# Arnault's matrix reshaping
index_matrix = np.array([[63, 38, 37, 12, 11, 63, 38, 37, 12, 11], # ankle
                [62, 39, 36, 13, 10, 62, 39, 36, 13, 10],
                [61, 40, 35, 14,  9, 61, 40, 35, 14,  9],
                [60, 41, 34, 15,  8, 60, 41, 34, 15,  8],
                [59, 42, 33, 16,  7, 59, 42, 33, 16,  7],
                [58, 43, 32, 17,  6, 58, 43, 32, 17,  6],
                [57, 44, 31, 18,  5, 57, 44, 31, 18,  5],
                [56, 45, 30, 19,  4, 56, 45, 30, 19,  4],
                [55, 46, 29, 20,  3, 55, 46, 29, 20,  3],
                [54, 47, 28, 21,  2, 54, 47, 28, 21,  2],
                [53, 48, 27, 22,  1, 53, 48, 27, 22,  1],
                [52, 49, 26, 23,  0, 52, 49, 26, 23,  0],
                [51, 50, 25, 24,  0, 51, 50, 25, 24,  0],
                [0, 24, 25, 50, 51,  0, 24, 25, 50, 51],
                [0, 23, 26, 49, 52,  0, 23, 26, 49, 52],
                [1, 22, 27, 48, 53,  1, 22, 27, 48, 53],
                [2, 21, 28, 47, 54,  2, 21, 28, 47, 54],
                [3, 20, 29, 46, 55,  3, 20, 29, 46, 55],
                [4, 19, 30, 45, 56,  4, 19, 30, 45, 56],
                [5, 18, 31, 44, 57,  5, 18, 31, 44, 57],
                [6, 17, 32, 43, 58,  6, 17, 32, 43, 58],
                [7, 16, 33, 42, 59,  7, 16, 33, 42, 59],
                [8, 15, 34, 41, 60,  8, 15, 34, 41, 60],
                [9, 14, 35, 40, 61,  9, 14, 35, 40, 61],
                [10, 13, 36, 39, 62, 10, 13, 36, 39, 62],
                [11, 12, 37, 38, 63, 11, 12, 37, 38, 63]]) # knee

# In the order of the cables, it is
# GRID 4    GRID 3
# GRID 1    GRID 2
# So taking the 256 signals in signal.data as input, one must reshape in
# the following way:

index_matrix[13:26,5:10] =  index_matrix[13:26,5:10] + 64 
index_matrix[0:13,5:10] = index_matrix[0:13,5:10] + 64 + 64 
index_matrix[0:13,0:5] = index_matrix[0:13,0:5] + 64 + 64 + 64   


if __name__ == '__main__':

    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/2mm'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    file = 'S1_25_2mm_Session1_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    Tx_max, Ty_max, theta_max  = 2, 2, 10*np.pi/180
    # Tx_max, Ty_max, theta_max  = 0,0,0
    print(os.listdir(DIR))
    signal, edition = open_mat_output(DIR, file)
    start, end = get_target_boundaries(signal['target'].squeeze())
    torch.set_default_dtype(torch.float64)

    # Apply filters to data and reshape into desired shape
    print('FILTER DATA...')
    emg = signal['data'][:, start:end]
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)
    emg_grid = make_grid(emg, index_matrix)
    H, W = emg_grid.shape[2], emg_grid.shape[3]
    Nch = H*W

    # Get separation matrix
    dts = edition['Dischargetimes']
    mu_dts = squeeze_dts(dts)
    mu_dts = filter_dts(mu_dts, start, end)

    R = 16
    sigma = 1.0
    tikhonov = 1e-4
    Tx, Ty, theta = -1.25, 1.25, 10*np.pi/180 # test integer shifts
    delta_width, delta_height = out_of_bounds_pixels(H, W, theta_max)
    xcrop, ycrop = Tx_max + floor(delta_width + 0.5), Ty_max + floor(delta_height + 0.5)

    # for rdx, R in enumerate([8, 16, 32, '1000/ch']):
    #     for sdx, sigma in enumerate([0.5, 1.0, 5.0]):
    #         for tikhonov in [1e-6, 1e-4, 1e-2]:

    # Apply gaussian blur to emg_grid
    emg_grid = gaussian_blur(emg_grid, kernel_size=3, sigma=1.0)

    # Crop observations and get new sep_mat
    print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
    emg_grid_test = apply_affine(emg_grid, Tx, Ty, theta) # Apply affine transformation and appropriate padding
    
    # Get minimum distance between original and transformed grid
    original_grid = get_transformed_grid((1, 1, H, W))
    transformed_grid = get_transformed_grid((1, 1, H, W), Tx, Ty, theta)
    center = transformed_grid[0:1, H//2:H//2 + 1, W//2:W//2 + 1, :]
    distances = torch.linalg.norm(original_grid - center, dim=3)
    min_distance = torch.min(distances)
    print(f'MIN DISTANCE: {min_distance} pixels')

    # Create mask based on transformed coordinates being within convex 
    transformed_coordinates = transformed_grid[0, :, :, :2].cpu().numpy().reshape(-1, 2)
    original_coordinates = original_grid[0, :, :, :2].cpu().numpy().reshape(-1, 2)
    hull = ConvexHull(transformed_coordinates)
    delaunay = Delaunay(transformed_coordinates[hull.vertices])
    inside = delaunay.find_simplex(original_coordinates) >= 0
    mask = torch.tensor(inside.reshape(H, W)).reshape(1, 1, H, W).float()

    # Get crop sep_mat
    if R == '1000/ch':
        R = 1000//(emg_grid_crop_train.shape[2]*emg_grid_crop_train.shape[3])
    extended_emg_crop_train = extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
    inv_cov = get_inv_cov_tikhonov(extended_emg_crop_train, reg=tikhonov)
    sep_mat_crop_train = get_sep_mat_torch(extended_emg_crop_train, mu_dts)
    sep_mat_crop_train = sep_mat_crop_train @ inv_cov

    # Initialize SDA module
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=sep_mat_crop_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
    base_loss = get_base_loss(emg_grid.to(torch.float64), sda, batch_size=batch_size, loss='kurtosis', device='cpu')

    # # Add optimal parameters for testing
    # with torch.no_grad():
    #     sda.sal.xshift.copy_(-2*Tx/W)
    #     sda.sal.yshift.copy_(-2*Ty/H)
    #     sda.sal.rot_theta.copy_(-theta/np.pi)

    with torch.no_grad():
        source_est_crop = sda(emg_grid_test)
    
    pred_dts, sils = get_silohuette(source_est_crop)
    matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)

    # # Test that masking channels is a valid solution
    # print('TESTING MASKING CHANNELS...')
    # R = 16
    # emg_grid_valid = emg_grid * mask
    # extended_emg_valid = extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # inv_cov_valid = get_inv_cov_tikhonov(extended_emg_valid, reg=tikhonov)
    # sep_mat_valid = get_sep_mat_torch(extended_emg_valid, mu_dts)
    # sep_mat_valid = sep_mat_valid @ inv_cov_valid
    # sda.xcrop, sda.ycrop = 0, 0
    # sda.extension_factor = R
    # sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)

    # with torch.no_grad():
    #     source_est_valid = sda(emg_grid_test)
    # pred_dts, sils = get_silohuette(source_est_valid)
    # matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    # print(f1_scores)

    # ## Test that separation matrix is working
    # source_est_crop = sep_mat_crop_train @ extended_emg_crop_train
    # spktrain = torch.zeros(source_est_crop.shape[0])
    # spktrain[mu_dts[1]] = 1.0
    # plt.figure()
    # plt.plot(source_est_crop[0,:1000]/source_est_crop[0,:1000].max())
    # plt.plot(spktrain[:1000])
    # # plt.plot(edition['Pulsetrain'][0,0][0,start:start+1000] / edition['Pulsetrain'][0,0][0,start:start+1000].max())
    # plt.legend(['Source Estimate', 'Pulse Trains'])
    # plt.savefig('test_sep_mat')

    # loss_arr = loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, T=(Tx, Ty), bounds=(2.5, 2.5), batch_size=batch_size, num_points=20, loss='kurtosis', device=device)
    losses = search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=50, lr=1e-3, boundaries=(2.5, 2.5, 10.0*np.pi/180), device='cuda')
    
    # Get new inverse covariance
    sda = sda.to('cpu')

    # Initial performance
    with torch.no_grad():
        sources = sda(emg_grid_test)
    pred_dts, sils = get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)

    # # Obtain new separation matrix with all valid channels
    # print('Obtaining new separation matrix...')
    # emg_grid_valid = emg_grid * mask # mask out channels that are out of bounds
    # extended_emg_valid = extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    # inv_cov_valid = get_inv_cov_tikhonov(extended_emg_valid, reg=tikhonov)
    # sep_mat_valid = get_sep_mat_torch(extended_emg_valid, mu_dts)
    # sep_mat_valid = sep_mat_valid @ inv_cov_valid

    # # Update SDA module with new separation matrix
    # sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
    # sda.xcrop, sda.ycrop = 0, 0
    #     # Get initial source estimates
    # with torch.no_grad():
    #     sources = sda(emg_grid_test)
    # pred_dts, sils = get_silohuette(sources)
    # matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    # print(f1_scores)

    # Get initial source estimates
    with torch.no_grad():
        sources = sda(emg_grid_test)
    pred_dts, sils = get_silohuette(sources)

    # Get new covariance matrix
    print('Getting new inverse covariance...')
    sda.xcrop, sda.ycrop = 0, 0
    extended_emg = sda.get_extended_emg(emg_grid_test)
    inv_cov = get_inv_cov_tikhonov(extended_emg.T, reg=tikhonov)
    # Refine separation matrix
    N = 150
    print('REFINING SEPARATION MATRIX...')
    with torch.no_grad():
        for idx in tqdm(range(N)):
            sda.refine_sep_mat(emg_grid_test, pred_dts, inv_cov)
            sources = sda(emg_grid_test)
            pred_dts, sils = get_silohuette(sources)

    pred_dts, sils = get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    print(f1_scores)

  
        