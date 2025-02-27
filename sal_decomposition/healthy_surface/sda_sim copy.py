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
from loss_functions import KurtosisLoss, NegentropyLoss
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay


# 2mm grid index matrix: make each subgrid individually, then concatenate them
grid1 = np.array([[48,40,43,44,45],
         [47,49,33,34,45], 
         [41,42,52,53,46],
         [51,50,54,35,36],
         [55,56,64,26,37],
         [63,62,61,38,27],
         [60,59,58,39,28],
         [57,1,2,29,19],
         [3,4,5,30,31],
         [6,7,8,32,20],
         [16,15,14,21,22],
         [11,12,13,23,24],
         [17,9,10,25,18]]) - 1

grid2 = np.array([[61,55,53,62,63],
         [61,56,54,52,64],
         [57,43,49,50,51],
         [60,58,44,42,41], # replaced 443 with 44
         [34,59,47,46,45],
         [25,33,39,40,48],
         [27,26,36,37,38],
         [1,28,30,29,35],
         [3,2,24,32,31],
         [16,14,21,22,23],
         [7,8,18,19,20],
         [5,6,10,9,17],
         [4,15,13,12,11]]) + 64 - 1 # replaced 14 with 4

grid3 = np.array([[62,53,52,51,50], # replaced 53 with 52
         [54,55,49,41,42],
         [56,64,43,44,45],
         [63,57,46,47,48],
         [58,59,40,39,38],
         [60,61,37,36,35],
         [34,33,29,30,31],
         [25,26,32,24,23],
         [27,28,22,21,20],
         [1,19,17,10,12],
         [2,18,9,11,13], # replaced 19 with 9
         [3,4,7,15,14],
         [3,6,8,5,16]]) + 64*2 - 1 # replaced 15 with 5 

grid4 = np.array([[42,41,49,43,40],
         [50,51,52,45,44],
         [55,54,53,47,46],
         [63,64,56,35,48],
         [60,61,62,37,36],
         [57,58,59,39,38],
         [3,2,1,33,34],
         [6,5,4,29,28],
         [16,8,7,31,30],
         [12,11,15,23,32],
         [9,17,13,14,22],
         [10,19,26,24,21],
         [20,18,27,25,21]]) + 64*3 - 1

# from collections import Counter
# counts = Counter(list(grid2.flatten()))
# dups = {key: counts[key] for key in counts.keys() if counts[key]>1}
# missed = set(list(range(1,65))).difference(list(grid2.flatten()))
# print()
index_matrix2 = np.vstack((np.hstack((grid2, grid1)), np.hstack((grid3, grid4))))
print()

                 

# Arnault's matrix reshaping
index_matrix4 = np.array([[63, 38, 37, 12, 11, 63, 38, 37, 12, 11], # ankle
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

index_matrix4[13:26,5:10] =  index_matrix4[13:26,5:10] + 64 
index_matrix4[0:13,5:10] = index_matrix4[0:13,5:10] + 64 + 64 
index_matrix4[0:13,0:5] = index_matrix4[0:13,0:5] + 64 + 64 + 64   

# Make random index matrix for comparison purposes
index_matrix_rand = np.r_[np.arange(256), np.array([0, 100, 121, 72])]
np.random.shuffle(index_matrix_rand)
index_matrix_rand = index_matrix_rand.reshape(26, 10)


def average_cv(spike_trains):
    """
    Computes the average coefficient of variation (CV) across motor unit spike trains.

    Parameters:
    - spike_trains (list of np.array): List where each element is an array of spike times (in samples) for a motor unit.

    Returns:
    - float: Average coefficient of variation (CV) across motor units.
    """
    cvs = []

    for spikes in spike_trains:
        if len(spikes) < 2:
            continue  # Skip motor units with fewer than 2 spikes

        # Compute inter-spike intervals (ISI)
        isi = np.diff(spikes)

        # Calculate coefficient of variation (std / mean)
        cv = np.std(isi) / np.mean(isi)
        cvs.append(cv)

    if not cvs:
        raise ValueError("No valid motor units with at least two spikes.")

    return np.mean(cvs)

def handle_outliers(emg_grid):
    '''Determine outlier channels, and replace them with average of neighbours.'''
    # Determine coordinates of outliers
    H, W = emg_grid.shape[2:]
    emg_grid_var = emg_grid.var(dim=[0,1])
    Q1, Q3 = torch.quantile(emg_grid_var.flatten(), 0.25), torch.quantile(emg_grid_var.flatten(), 0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 -3.0*IQR, Q3 + 3.0*IQR
    y, x = torch.where(torch.logical_or(emg_grid_var >= upper, emg_grid_var <= lower)) # only keep non-noisy channel
    y, x = y.tolist(), x.tolist()

    idx = 0
    while idx < len(y): # for each outlier
        l,r,b,t = x[idx] != 0, x[idx] != W-1, y[idx] != H-1, y[idx] != 0
        subgrid = emg_grid[:, :, y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten(start_dim=2, end_dim=3)
        subgridvar = emg_grid_var[y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten()
        subgrid = subgrid[:, :, torch.logical_and(subgridvar < upper, subgridvar > lower)] # remove outlier channels included
        if subgrid.shape[2] < 3: # if less than 3 valid neighbours, try again after filling in more channels
            y.append(y[idx])
            x.append(x[idx])
        else:
            emg_grid[:,:,y[idx], x[idx]] = subgrid.mean(dim=2) # compute as average of neighbours
        idx += 1

    return emg_grid


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
    # Tx_max, Ty_max,   
    # _max  = 0,0,0
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

    # Zero out outlier channels
    # emg_grid_var = emg_grid.var(dim=[0,1])
    # Q1, Q3 = torch.quantile(emg_grid_var.flatten(), 0.25), torch.quantile(emg_grid_var.flatten(), 0.75)
    # IQR = Q3 - Q1
    # threshold = Q3 + 1.5*IQR
    # emg_grid[:, :, emg_grid_var > threshold] = 0.0 # set noisy channels to zero

    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    emg_grid = handle_outliers(emg_grid)
    emg_grid = emg_grid / (emg_grid.std() + 1e-12)

    # Get separation matrix
    dts = edition['Dischargetimes']
    mu_dts = utils.squeeze_dts(dts)
    mu_dts = utils.filter_dts(mu_dts, start, end)

    R = 16
    # tikhonov = 1e-2
    explained_var = 1-1e-3
    Tx, Ty, theta = -1.25, 0.75, 0.0 #10*np.pi/180 # test integer shift
    delta_width, delta_height = utils.out_of_bounds_pixels(H, W, theta_max)
    xcrop, ycrop = Tx_max + floor(delta_width + 0.5), Ty_max + floor(delta_height + 0.5)

    # for rdx, R in enumerate([8, 16, 32, '1000/ch']):
    #     for tikhonov in [1e-6, 1e-4, 1e-2]:

    # Apply gaussian blur to emg_grid
    # emg_grid = gaussian_blur(emg_grid, kernel_size=3, sigma=sigma)

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
    transformed_coordinates = transformed_grid[0, :, :, :2].cpu().numpy().reshape(-1, 2)
    original_coordinates = original_grid[0, :, :, :2].cpu().numpy().reshape(-1, 2)
    hull = ConvexHull(transformed_coordinates)
    delaunay = Delaunay(transformed_coordinates[hull.vertices])
    inside = delaunay.find_simplex(original_coordinates) >= 0
    mask = torch.tensor(inside.reshape(H, W))
    
    # Find most conservative crop
    lcrop, rcrop, bcrop, tcrop = W//2 - 1, W//2 - 1, H//2 - 1, H//2 - 1
    min_crop = False
    while not min_crop:
        crop_sum = lcrop + rcrop + bcrop + tcrop
        if mask[tcrop:H-bcrop, lcrop-1:W-rcrop].all() and lcrop > 0:
            lcrop -= 1
        if mask[tcrop:H-bcrop, lcrop:W-(rcrop-1)].all() and rcrop > 0:
            rcrop -= 1
        if mask[tcrop-1:H-bcrop, lcrop:W-rcrop].all() and tcrop > 0:
            tcrop -= 1
        if mask[tcrop:H-(bcrop-1), lcrop:W-rcrop].all() and bcrop > 0:
            bcrop -= 1
        if crop_sum == lcrop + rcrop + bcrop + tcrop: # if no more changes, we have found the minimum crop
            min_crop = True


    # Test that masking channels is a valid solution
    print('TESTING MASKING CHANNELS...')
    emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
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
    # loss_arr = loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, T=(Tx, Ty), bounds=(2.0, 2.0), batch_size=batch_size, num_points=20, loss='kurtosis', device=device)
    losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=50, lr=5e-3, boundaries=(2.5, 2.5, 10.0*np.pi/180), device='cuda')
    
    # Get new inverse covariance
    sda = sda.to('cpu')
    sda.sal.mode = 'bicubic'

    # # Initial performance
    # with torch.no_grad():
    #     sources = sda(emg_grid_test)
    # pred_dts, sils = get_silohuette(sources)
    # matches, f1_scores, sensitivities, precisions = spike_matching(mu_dts, pred_dts, fs=fsamp)
    # print(f1_scores)

    # Obtain new separation matrix with all valid channels
    print('Obtaining new separation matrix...')
    sda.lcrop, sda.rcrop = lcrop, rcrop
    sda.bcrop, sda.tcrop = bcrop, tcrop
    emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
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

    # # Get initial source estimates
    # with torch.no_grad():
    #     sources = sda(emg_grid_test)
    # pred_dts, sils = get_silohuette(sources)

    # Get new covariance matrix
    print('Getting new inverse covariance...')
    sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
    # extended_emg_sal = sda.get_extended_emg(emg_grid_test).T
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    # inv_cov_test = get_inv_cov_tikhonov(extended_emg_sal, reg=1e-12)
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

  
        