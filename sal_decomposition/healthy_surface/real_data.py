import numpy as np
import scipy
from scipy.io import loadmat
from scipy import signal
import os
import json
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

def make_spike_train(dts, T):
    spike_train = np.zeros((len(dts), T))
    for dt_idx, dt in enumerate(dts):
        spike_train[dt_idx, dt] = 1.0
    return spike_train

def bandstop_filter(data, fsamp=2048): ## ACTUALLY BANDSTOP, NOT NOTCH
    '''Used to remove powerline interference and its multiples.'''
    sos = signal.butter(4, (45, 55), btype='bandstop', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def bandpass_filter(data, fsamp=2048):
    '''Used to maintain only information in relevant anatomical range of sEMG activity.'''
    sos = signal.butter(2, (20, 500), btype='bandpass', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def notch_filter(data, fsamp=2048):
    '''Used to remove powerline interference at 50Hz.'''
    b, a = signal.iirnotch(50, Q=30, fs=fsamp)
    sos = signal.tf2sos(b, a)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


if __name__ == '__main__':

    # Parameters
    subject = 3
    ied = 4
    mvc = 25
    session1 = 2
    session2 = 3

    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited'
    DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject}_edited'
    index_matrix = index_matrix4

    if subject == 1:
        DIR = os.path.join(DIR, f'{ied}mm')        

    if ied == 2: 
        mvc = f"{mvc}_2mm"
        index_matrix = index_matrix2

    with open('./sal_decomposition/healthy_surface/outlier_channels.json', 'r') as f:
        outliers = json.load(f) 
        
    file = f'S{subject}_{mvc}_Session{session1}_MUEdit_edited.mat'
    file2 = f'S{subject}_{mvc}_Session{session2}_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    bounds = [2.5, 2.5, 10*np.pi/180]

    delta_width, delta_height = utils.out_of_bounds_pixels(26, 10, bounds[2])
    xcrop, ycrop = int(bounds[0] + floor(delta_width + 0.5)), int(bounds[1] + floor(delta_height + 0.5))

    # Load training data
    sgnl, edition = utils.open_mat_output(DIR, file)
    start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
    print('FILTERING TRAINING DATA...')
    emg = sgnl['data']
    emg = emg - emg.mean(axis=0, keepdims=True) # average referencing
    emg = bandstop_filter(bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg_grid_original = utils.make_grid(emg, index_matrix)
    H, W = emg_grid_original.shape[2], emg_grid_original.shape[3]
    Nch = H*W
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    visible_outliers = np.zeros((H, W), dtype=np.bool_)
    outlier_coords = outliers[f"subject{subject}"][f"session{session1}"]
    indices = tuple(np.array(outlier_coords).T)
    visible_outliers[indices] = True
    emg_grid_original = utils.handle_outliers(emg_grid_original, visible_outliers=visible_outliers)
    emg_grid = emg_grid_original[start:end]

    # Load discharge times
    dts = edition['Dischargetimes']
    mu_dts = utils.squeeze_dts(dts)
    mu_dts = utils.filter_dts(mu_dts, start, end)
    
    # Load test data
    sgnl2, edition2 = utils.open_mat_output(DIR, file2)
    start2, end2 = utils.get_target_boundaries(sgnl2['target'].squeeze())
    emg2 = sgnl2['data']
    emg2 = emg2 - emg2.mean(axis=0, keepdims=True) # average referencing

    print('FILTERING TEST DATA...')
    emg2 = bandstop_filter(bandpass_filter(emg2, fsamp=fsamp), fsamp=fsamp)
    # emg2 = bandpass_filter(emg2, fsamp=fsamp)
    emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12)
    emg_grid_test_original = utils.make_grid(emg2, index_matrix)
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    visible_outliers = np.zeros((H, W), dtype=np.bool_)
    outlier_coords = outliers[f"subject{subject}"][f"session{session2}"]
    indices = tuple(np.array(outlier_coords).T)
    visible_outliers[indices] = True
    emg_grid_test_original = utils.handle_outliers(emg_grid_test_original, visible_outliers=visible_outliers)
    emg_grid_test = emg_grid_test_original[start2:end2]

    # Load discharge times
    dts2 = edition2['Dischargetimes']
    mu_dts2 = utils.squeeze_dts(dts2)
    mu_dts2 = utils.filter_dts(mu_dts2, start2, end2)

    # Set pipeline parameters
    R = 16
    reg = 1e-1

    # Get crop sep_mat
    print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
    extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
    inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop_train, reg=reg).to(torch.float32)
    # emg_grid_crop = emg_grid[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
    # extended_emg = utils.extend_emg_torch(emg_grid_crop.squeeze().reshape(emg_grid_crop.shape[0], -1), R).T
    # inv_cov_train = utils.get_inv_cov_torch(extended_emg, explained_var=explained_var).to(torch.float32)
    print('STA template estimation with peeling...')
    extended_emg = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
    STA = utils.get_sta_templates(extended_emg, mu_dts).to(torch.float32) # gets STA templates from Session 1 from the raw data
    # STA = utils.get_sta_templates_peeloff(emg_grid, mu_dts, R=R, L=50).to(torch.float32)

    # Initialize SDA module
    # sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
    # sda.sal.mode = 'bicubic'

    with torch.no_grad():
        sources = sda(emg_grid)
    pred_dts_train, sils = utils.get_silohuette(sources)
    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts_train, fs=fsamp)
    cov_isi = utils.get_trimmed_cov_isi(pred_dts_train)
    print('RoA Training:', np.mean(rate_of_agreement))

    # Get top MU candidates
    sorted_idxs = np.argsort(rate_of_agreement)[::-1]
    mu_dts = [mu_dts[idx] for idx in sorted_idxs]
    STA = utils.get_sta_templates_peeloff(emg_grid, mu_dts, R=R, L=50).to(torch.float32)
    sda.STA = STA

    # Get base loss so we can understand how much sparsity relative to the training set/original decomposition
    base_loss = utils.get_base_loss(emg_grid, sda.to(device), batch_size=batch_size, loss='negentropy', device=device)
    
    # Get inverse covariance of the test grid, and determine the spatial transformation required for the STA templates to become optimal separation vectors
    emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
    extended_emg = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
    inv_cov = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
    sda.inv_cov = inv_cov

    # loss_arr = utils.loss_sampling(emg_grid_test.clone(), sda.to(device), base_loss=base_loss, bounds=bounds[:2], batch_size=batch_size, num_points=20, loss='negentropy', device=device)
    sources, losses = utils.search_fit_sda(emg_grid_test.clone(), sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=100, lr=5e-3, boundaries=bounds, loss='negentropy', device=device)

    # Get new inverse covariance
    sda = sda.to('cpu')
    sda.sal.mode = 'bicubic'

    # Get minimum distance between original and transformed grid
    Tx, Ty, theta = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2, sda.sal.rot_theta[0].item()*np.pi
    print(f"Tx: {Tx}, Ty: {Ty}, theta: {theta}")

    # original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), -Tx, -Ty, -theta)
    # print(f'MIN DISTANCE: {min_distance} pixels')

    # Create mask based on transformed coordinates being within convex 
    # lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

    # Update SDA module with new separation matrix
    print('Obtaining new separation matrix...')
    # Create mask based on transformed coordinates being within convex
    print('RECOMPUTING MINIMALLY CROPPED INVERSE COVARIANCE MATRIX...')
    theta = sda.sal.get_affine_transform(input_shape=(H, W), inverse=True)
    theta = theta.repeat(1, 1, 1)
    transformed_grid = sda.sal.get_grid(theta, input_shape=(H, W))
    transformed_grid[:,:,:,0] = (W-1)*(1 + transformed_grid[:,:,:,0])/2
    transformed_grid[:,:,:,1] = (H-1)*(1 + transformed_grid[:,:,:,1])/2

    original_grid = torch.nn.functional.affine_grid(torch.eye(3)[0:2,:].unsqueeze(0), size=(1,1,H,W), align_corners=True)
    original_grid[:,:,:,0] = (W-1)*(1 + original_grid[:,:,:,0])/2
    original_grid[:,:,:,1] = (H-1)*(1 + original_grid[:,:,:,1])/2

    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid.detach().cpu(), original_grid)
    sda.lcrop, sda.rcrop, sda.tcrop, sda.bcrop = lcrop, rcrop, tcrop, bcrop
    sda.crop_mask = sda.get_crop_mask()

    emg_grid_crop_test = emg_grid_test[:, :, tcrop:emg_grid_test.shape[2]-bcrop, lcrop:emg_grid_test.shape[3]-rcrop].clone()
    extended_emg = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
    sda.inv_cov = inv_cov_test

    # Get initial source estimates
    with torch.no_grad():
        sources = sda(emg_grid_test)

    # Get performance on new test grid post training
    pred_dts, sils = utils.get_silohuette(sources)
    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts2, pred_dts, fs=fsamp)
    fr_pred = utils.get_mean_firing_rate(pred_dts, fs=fsamp)
    print(rate_of_agreement)

    match_data = []
    for idx, (pred_idx, true_idx) in enumerate(matches.items()):
        # if zscores[pred_idx] > 3:
        # Bundle everything together
        match_info = {
            "mu_dts": mu_dts[pred_idx],
            "pred_dts": pred_dts[pred_idx],
            "mu_dts2": mu_dts2[true_idx],
            "sil": sils[pred_idx],
            "roa": rate_of_agreement[idx]
        }
        match_data.append(match_info)

    # Sort by SIL
    match_data = sorted(match_data, key=lambda x: x['roa'], reverse=True)

    # # Determine which MUs are above the threhsold and constitute matches
    # filtered_matches = {}
    # filtered_pred_dts, filtered_mu_dts, filtered_sils = [], [], []
    # counter = 0
    # for idx, key in enumerate(matches.keys()):
    #     if zscores[idx] > 3:
    #         filtered_matches[counter] = matches[key]
    #         filtered_pred_dts.append(pred_dts[key])
    #         filtered_mu_dts.append(mu_dts[key])
    #         filtered_sils.append(sils[key])
    #         counter += 1

    # mu_dts = filtered_mu_dts.copy()
    # pred_dts = filtered_pred_dts.copy()
    # sils = filtered_sils.copy()
    # matches = dict(filtered_matches)
    print(f"#Matches: {len(match_data)}")

    # # Now we sort the matched MUs by SIL
    # sorted_idxs = np.argsort(sils)[::-1]
    # mu_dts = [mu_dts[idx] for idx in sorted_idxs]
    # pred_dts = [pred_dts[idx] for idx in sorted_idxs]
    # _matches = dict(matches)
    # matches = {idx: _matches[sort_idx] for idx, sort_idx in enumerate(sorted_idxs)}

    # Get new covariance matrix 
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T.to(torch.float32)
    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg_test, reg=1e-3).to(torch.float32)


    # Refinement of MUs detected
    Nr = 20
    # official_matches = dict(matches)
    for idx in tqdm(range(Nr)):
        # Get sorted pred_dts
        pred_dts = [item['pred_dts'] for item in match_data]

        with torch.no_grad():
            # sta_test = utils.get_sta_templates_peeloff(emg_grid_test.clone(), pred_dts, R=R, L=50).to(torch.float32)
            sta_test = utils.get_sta_templates(extended_emg_test, pred_dts).to(torch.float32)
            sep_mat_test = sta_test @ inv_cov_test
            sources = (sep_mat_test @ extended_emg_test).T

        # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
        pred_dts, sils = utils.get_silohuette(sources)
        for i, item in enumerate(match_data):
            item['pred_dts'] = pred_dts[i]
            item['sil'] = sils[i]

        mu_dts2 = [item['mu_dts2'] for item in match_data]
        mu_dts = [item['mu_dts'] for item in match_data]

        # Compute metrics
        matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts2, pred_dts, old_matches={idx:idx for idx in range(len(pred_dts))}, fs=fsamp)
        fr_train, fr_pred = utils.get_median_firing_rate(mu_dts, fs=fsamp), utils.get_median_firing_rate(pred_dts, fs=fsamp)
        # corrs = [utils.get_sta_correlations(emg_grid_test.clone(), pred_dts[idx].astype(int), mu_dts2[idx].astype(int), L=50) for idx in range(len(pred_dts))]
        print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.7 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")
        print(f"Average firing rate difference between matches: {np.mean([np.abs(fr_train[idx]-fr_pred[idx]) for idx in range(len(mu_dts)) if rate_of_agreement[idx] > 0.7])}")

        # Adding RoA to match data
        for i, item in enumerate(match_data):
            item['roa'] = rate_of_agreement[i]

        # Sort again based on SIL value
        match_data = sorted(match_data, key=lambda x: x['roa'], reverse=True)

    print(rate_of_agreement)
    print()

