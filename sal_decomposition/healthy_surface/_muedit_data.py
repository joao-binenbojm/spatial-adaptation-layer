import numpy as np
import scipy
from scipy.io import loadmat
from scipy import signal
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


def bandstop_filter(data, fsamp=2048): ## ACTUALLY BANDSTOP, NOT NOTCH
    '''Used to remove powerline interference and its multiples.'''
    sos = signal.butter(4, (45, 55), btype='bandstop', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def bandpass_filter(data, fsamp=2048):
    '''Used to maintain only information in relevant anatomical range of sEMG activity.'''
    sos = signal.butter(4, (20, 500), btype='bandpass', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def notch_filter(data, fsamp=2048):
    '''Used to remove powerline interference at 50Hz.'''
    b, a = signal.iirnotch(50, Q=30, fs=fsamp)
    sos = signal.tf2sos(b, a)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


if __name__ == '__main__':

    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited'
    DIR = f'/home/joao/Desktop/datasets/muedit_data/'
    file = "S1_20_DF.otb+_decomp.mat_edited.mat"
    index_matrix = index_matrix4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    bounds = [3, 3, 10*np.pi/180]
    # torch.set_default_dtype(torch.float64)
    print(os.listdir(DIR))

    # Load training data
    sgnl, edition = utils.open_mat_output(DIR, file)
    start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
    print('FILTERING TRAINING DATA...')
    emg = sgnl['data'][:256, start:end]
    # emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    # emg = bandstop_filter(bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
    emg = bandpass_filter(emg, fsamp=fsamp)
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg_grid = utils.make_grid(emg, index_matrix)
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
    
    emg_grid[:, 0, 0:13, 0:5] = emg_grid[:, 0, 0:13, 0:5] / (emg_grid[:, 0, 0:13, 0:5].std() + 1e-12)
    emg_grid[:, 0, 0:13, 5:10] = emg_grid[:, 0, 0:13, 5:10]/ (emg_grid[:, 0, 0:13, 5:10].std() + 1e-12)
    emg_grid[:, 0, 13:26, 0:5] = emg_grid[:, 0, 13:26, 0:5] / (emg_grid[:, 0, 13:26, 0:5].std() + 1e-12)
    emg_grid[:, 0, 13:26, 5:10] = emg_grid[:, 0, 13:26, 5:10] / (emg_grid[:, 0, 13:26, 5:10].std() + 1e-12)

    # Align MUAPs in a consistent way across sessions, to make comparison via discharge times more straightforward
    # mu_dts, shifts = utils.get_aligned_discharge_times(emg_grid, mu_dts, L=100, energy_threshold=0.5)

    # Load test data
    sgnl2, edition2 = utils.open_mat_output(DIR, file2)
    start2, end2 = utils.get_target_boundaries(sgnl2['target'].squeeze())
    emg2 = sgnl2['data'][:, start2:end2]
    # emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12) # centering emg
    print('FILTERING TEST DATA...')
    emg2 = bandstop_filter(bandpass_filter(emg2, fsamp=fsamp), fsamp=fsamp)
    # emg2 = bandpass_filter(emg2, fsamp=fsamp)
    emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12)
    emg_grid_test = utils.make_grid(emg2, index_matrix)
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    emg_grid_test = utils.handle_outliers(emg_grid_test)

    # Normalize each subgrid independently
    # if ied == 4:
    emg_grid[:, 0, 0:13, 0:5] = emg_grid[:, 0, 0:13, 0:5] / (emg_grid[:, 0, 0:13, 0:5].std() + 1e-12)
    emg_grid[:, 0, 0:13, 5:10] = emg_grid[:, 0, 0:13, 5:10]/ (emg_grid[:, 0, 0:13, 5:10].std() + 1e-12)
    emg_grid[:, 0, 13:26, 0:5] = emg_grid[:, 0, 13:26, 0:5] / (emg_grid[:, 0, 13:26, 0:5].std() + 1e-12)
    emg_grid[:, 0, 13:26, 5:10] = emg_grid[:, 0, 13:26, 5:10] / (emg_grid[:, 0, 13:26, 5:10].std() + 1e-12)

    emg_grid_test[:, 0, 0:13, 0:5] = emg_grid_test[:, 0, 0:13, 0:5] / (emg_grid_test[:, 0, 0:13, 0:5].std() + 1e-12)
    emg_grid_test[:, 0, 0:13, 5:10] = emg_grid_test[:, 0, 0:13, 5:10]/ (emg_grid_test[:, 0, 0:13, 5:10].std() + 1e-12)
    emg_grid_test[:, 0, 13:26, 0:5] = emg_grid_test[:, 0, 13:26, 0:5] / (emg_grid_test[:, 0, 13:26, 0:5].std() + 1e-12)
    emg_grid_test[:, 0, 13:26, 5:10] = emg_grid_test[:, 0, 13:26, 5:10] / (emg_grid_test[:, 0, 13:26, 5:10].std() + 1e-12)

    # Load discharge times
    dts2 = edition2['Dischargetimes']
    mu_dts2 = utils.squeeze_dts(dts2)
    mu_dts2 = utils.filter_dts(mu_dts2, start2, end2)

    # Align MUAPs in a consistent way across sessions, to make comparison via discharge times more straightforward
    # mu_dts2, shifts2 = utils.get_aligned_discharge_times(emg_grid_test, mu_dts2, L=100, energy_threshold=0.5)

    # Set pipeline parameters
    R = 16
    explained_var = 1-1e-4
    delta_width, delta_height = utils.out_of_bounds_pixels(H, W, 0.0)
    xcrop, ycrop = bounds[0] + floor(delta_width + 0.5), bounds[1] + floor(delta_height + 0.5)

    # Crop observations and get new sep_mat
    # print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    # emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()

    # Get crop sep_mat
    extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
    inv_cov_train = utils.get_inv_cov_torch(extended_emg_train, explained_var=explained_var).to(torch.float32)
    # STA = utils.get_sta_templates(extended_emg_train, mu_dts).to(torch.float32)
    print('STA template estimation with peeling...')
    STA = utils.get_sta_templates_peeloff(emg_grid, mu_dts, R=R, L=50).to(torch.float32)

    # Initialize SDA module
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, extension_factor=R)
    # sda.sal.mode = 'bicubic'

    with torch.no_grad():
        sources = sda(emg_grid)
    pred_dts_train, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts_train, fs=fsamp)
    print('F1-Score Training:', np.mean(f1_scores))

    # Get base loss so we can understand how much sparsity relative to the training set/original decomposition
    base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss='kurtosis', device='cpu')
    
    # Get inverse covariance of the test grid, and determine the spatial transformation required for the STA templates to become optimal separation vectors
    sda.lcrop, sda.rcrop = xcrop, xcrop
    sda.tcrop, sda.bcrop = ycrop, ycrop
    sda.crop_mask = sda.get_crop_mask()
    emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
    extended_emg_test = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=explained_var).to(torch.float32)
    sda.inv_cov = inv_cov_test

    sources, losses = utils.search_fit_sda(emg_grid_test.clone(), sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=1000, nepochs=0, lr=5e-4, boundaries=bounds, device='cuda')
    
    # Get new inverse covariance
    sda = sda.to('cpu')
    # sda.sal.mode = 'bicubic'

    # Get minimum distance between original and transformed grid
    Tx, Ty, theta = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2, sda.sal.rot_theta[0].item()*np.pi
    original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
    print(f'MIN DISTANCE: {min_distance} pixels')
    print(f"Tx: {Tx}, Ty: {Ty}, theta: {theta}")

    # Create mask based on transformed coordinates being within convex 
    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

    # Update SDA module with new separation matrix
    print('Obtaining new separation matrix...')
    sda.lcrop, sda.rcrop = lcrop, rcrop
    sda.bcrop, sda.tcrop = bcrop, tcrop
    emg_grid_valid = emg_grid_test[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
    extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
    inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var).to(torch.float32)
    sda.inv_cov = inv_cov_valid
    sda.crop_mask = sda.get_crop_mask()

    # Get initial source estimates
    with torch.no_grad():
        sources = sda(emg_grid_test)

    # Get performance on new test grid post training
    pred_dts, sils = utils.get_silohuette(sources)
    matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts2, pred_dts, fs=fsamp)
    print(f1_scores)


    # def make_spikes(T, N, fs=2048, freq=15, cov=0.3):
    #     '''Make random spikes so we can test whether a refinement can lead to matches. '''
    #     isi = int(fs/freq)
    #     is_std = isi*cov

    #     pred_dts = []
    #     for n in range(N):
    #         dts = []
    #         t = 0
    #         while t < T:
    #             interval = np.random.normal(loc=isi, scale=is_std)
    #             t += int(interval)
    #             if t >= T:
    #                 continue
    #             else:
    #                 dts.append(t)
    #         pred_dts.append(np.array(dts))
    #     return pred_dts

    # pred_dts = make_spikes(emg_grid_test.shape[0], len(mu_dts))

    # Get new covariance matrix
    print('Refining MUAP shapes based on predicted dts...')
    sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=explained_var)

    # Refinement of MUs detected
    Nr = 30
    for idx in tqdm(range(Nr)):
        with torch.no_grad():
            sta_test = utils.get_sta_templates_peeloff(emg_grid_test, pred_dts, R=R, L=40)
            sep_mat_test = sta_test @ inv_cov_test
            sources = (sep_mat_test @ extended_emg_test).T

        pred_dts, sils = utils.get_silohuette(sources)
        matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts2, pred_dts, fs=fsamp)
        print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([f > 0.8 for f in f1_scores])}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")
    
    print(f1_scores)

