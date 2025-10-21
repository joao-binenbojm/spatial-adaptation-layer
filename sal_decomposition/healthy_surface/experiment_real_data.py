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

# def notch_filter(data, fsamp=2048):
#     '''Used to remove powerline interference at 50Hz.'''
#     b, a = signal.iirnotch(50, Q=30, fs=fsamp)
#     sos = signal.tf2sos(b, a)
#     data = signal.sosfiltfilt(sos, data, axis=1)
#     return data


if __name__ == '__main__':

    # Parameters
    subject = 3
    ied = 4
    mvc = 25
    DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject}_edited'
    index_matrix = index_matrix4

    if subject == 1:
        DIR = os.path.join(DIR, f'{ied}mm')        

    if ied == 2: 
        mvc = f"{mvc}_2mm"
        index_matrix = index_matrix2

    with open('./sal_decomposition/healthy_surface/outlier_channels.json', 'r') as f:
        outliers = json.load(f) 


    for idx, session_pairs in enumerate([[1,2], [2,3], [1,3]]):
        for jdx in range(2): # which way we adapt from
            session1 = jdx + 1
            session2 = 2 - jdx
        
            file = f'S{subject}_{mvc}_Session{session1}_MUEdit_edited.mat'
            file2 = f'S{subject}_{mvc}_Session{session2}_MUEdit_edited.mat'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fsamp = 2048
            batch_size = 10000
            bounds = [3, 3]#, 20*np.pi/180]

            # Load training data
            sgnl, edition = utils.open_mat_output(DIR, file)
            start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
            print('FILTERING TRAINING DATA...')
            emg = sgnl['data']
            # emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
            emg = bandstop_filter(bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
            # emg = bandpass_filter(emg, fsamp=fsamp)
            emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std(axis=1, keepdims=True) + 1e-12) # centering emg
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
            # emg_grid = emg_grid / (emg_grid.std() + 1e-12)
            # Load discharge times
            dts = edition['Dischargetimes']
            mu_dts = utils.squeeze_dts(dts)
            mu_dts = utils.filter_dts(mu_dts, start, end)
    
            # Align MUAPs in a consistent way across sessions, to make comparison via discharge times more straightforward
            # mu_dts, shifts = utils.get_aligned_discharge_times(emg_grid, mu_dts, L=100, energy_threshold=0.5)

            # Load test data
            sgnl2, edition2 = utils.open_mat_output(DIR, file2)
            start2, end2 = utils.get_target_boundaries(sgnl2['target'].squeeze())
            emg2 = sgnl2['data']
            # emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12) # centering emg
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
            fr_test = utils.get_mean_firing_rate(mu_dts2, fs=fsamp)

            # Align MUAPs in a consistent way across sessions, to make comparison via discharge times more straightforward
            # mu_dts2, shifts2 = utils.get_aligned_discharge_times(emg_grid_test, mu_dts2, L=100, energy_threshold=0.5)

            # Set pipeline parameters
            R = 16
            explained_var = 1-1e-2
            delta_width, delta_height = utils.out_of_bounds_pixels(H, W, 0.0)
            xcrop, ycrop = bounds[0] + floor(delta_width + 0.5), bounds[1] + floor(delta_height + 0.5)

            # # Estimate shifts, and apply them to mu_dts
            # print('UPDATING DISCHARGE TIMES TO BE CENTERED ABOUT PEAK ENERGY OF MUAP...')
            # shifts = []
            # L = 50

            # with torch.no_grad():
            #     for idx in tqdm(range(len(mu_dts))):
            #         sta = utils.get_sta_muaps(emg_grid, mu_dts[idx].astype(int), L=L, spacing=15, plot=False)
            #         peak = np.argmax((sta**2).mean(dim=[1,2])) - L
            #         shift = peak - R//2 # ensures our separation vector is centered about the energy peak
            #         mu_dts[idx] = mu_dts[idx] + shift.item()
            #         mu_dts[idx] = mu_dts[idx][np.logical_and(mu_dts[idx] >= 0, mu_dts[idx] < emg_grid.shape[0])] # ensures all shifted discharge times are valid

            #     for idx in tqdm(range(len(mu_dts2))):
            #         sta = utils.get_sta_muaps(emg_grid_test, mu_dts2[idx].astype(int), L=L, spacing=15, plot=False)
            #         peak = np.argmax((sta**2).mean(dim=[1,2])) - L
            #         shift = peak - R//2 # ensures our separation vector is centered about the energy peak
            #         mu_dts2[idx] = mu_dts2[idx] + shift.item()
            #         mu_dts2[idx] = mu_dts2[idx][np.logical_and(mu_dts2[idx] >= 0, mu_dts2[idx] < emg_grid_test.shape[0])] # ensures all shifted discharge times are valid

            # Get crop sep_mat
            # extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
            # inv_cov_train = utils.get_inv_cov_torch(extended_emg_train, explained_var=explained_var).to(torch.float32)
            emg_grid_crop = emg_grid[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
            extended_emg = utils.extend_emg_torch(emg_grid_crop.squeeze().reshape(emg_grid_crop.shape[0], -1), R).T
            inv_cov_train = utils.get_inv_cov_torch(extended_emg, explained_var=explained_var).to(torch.float32)
            print('STA template estimation with peeling...')
            STA = utils.get_sta_templates_peeloff(emg_grid, mu_dts, R=R, L=50).to(torch.float32)

            # Initialize SDA module
            sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
            # sda.sal.mode = 'bicubic'

            with torch.no_grad():
                sources = sda(emg_grid)
            pred_dts_train, sils = utils.get_silohuette(sources)
            matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts_train, fs=fsamp)
            print('RoA Training:', np.mean(rate_of_agreement))

            # Get top MU candidates
            corrs = [utils.get_sta_correlations(emg_grid, pred_dts_train[idx], mu_dts[idx], L=50) for idx in range(len(mu_dts))]
            sorted_idxs = np.argsort(sils)[::-1]
            mu_dts = [mu_dts[idx] for idx in sorted_idxs]
            STA = utils.get_sta_templates_peeloff(emg_grid, mu_dts[:10], R=R, L=50).to(torch.float32)
            sda.STA = STA

            # Get base loss so we can understand how much sparsity relative to the training set/original decomposition
            base_loss = utils.get_base_loss(emg_grid, sda.to(device), batch_size=batch_size, loss='negentropy', device=device)
            
            # Get inverse covariance of the test grid, and determine the spatial transformation required for the STA templates to become optimal separation vectors
            emg_grid_crop = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
            extended_emg = utils.extend_emg_torch(emg_grid_crop.squeeze().reshape(emg_grid_crop.shape[0], -1), R).T
            inv_cov = utils.get_inv_cov_torch(extended_emg, explained_var=explained_var).to(torch.float32)
            sda.inv_cov = inv_cov

            loss_arr = utils.loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, bounds=(3.0, 3.0), batch_size=batch_size, num_points=20, loss='negentropy', device=device)
            sources, losses = utils.search_fit_sda(emg_grid_test.clone(), sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=100, lr=5e-4, boundaries=bounds, loss='negentropy', device=device)

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
            emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
            extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
            inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var).to(torch.float32)
            sda.inv_cov = inv_cov_valid
            sda.crop_mask = sda.get_crop_mask()

            # Get predictions for all MUs now
            STA = utils.get_sta_templates_peeloff(emg_grid.clone(), mu_dts, R=R, L=50).to(torch.float32)
            sda.STA = STA

            # Get initial source estimates
            with torch.no_grad():
                sources = sda(emg_grid_test)


            # Get performance on new test grid post training
            pred_dts, sils = utils.get_silohuette(sources)
            matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts2, pred_dts, fs=fsamp)
            fr_pred = utils.get_mean_firing_rate(pred_dts, fs=fsamp)
            print(rate_of_agreement)
            print()

            # Determine which MUs are above the threhsold and constitute matches
            filtered_matches = {}
            filtered_pred_dts, filtered_mu_dts, filtered_sils = [], [], []
            counter = 0
            for idx, key in enumerate(matches.keys()):
                if zscores[idx] > 4:
                    filtered_matches[counter] = matches[key]
                    filtered_pred_dts.append(pred_dts[key])
                    filtered_mu_dts.append(mu_dts[key])
                    filtered_sils.append(sils[key])
                    counter += 1

            mu_dts = filtered_mu_dts.copy()
            pred_dts = filtered_pred_dts.copy()
            sils = filtered_sils.copy()
            matches = dict(filtered_matches)
            print(f"#Matches: {len(mu_dts)}")

            # Now we sort the matched MUs by SIL
            sorted_idxs = np.argsort(sils)[::-1]
            mu_dts = [mu_dts[idx] for idx in sorted_idxs]
            pred_dts = [pred_dts[idx] for idx in sorted_idxs]
            _matches = dict(matches)
            matches = {idx: _matches[sort_idx] for idx, sort_idx in enumerate(sorted_idxs)}

            # Get new covariance matrix 
            extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
            inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=1-1e-4)


            # Refinement of MUs detected
            Nr = 20
            official_matches = dict(matches)
            for idx in tqdm(range(Nr)):
                with torch.no_grad():
                    sta_test = utils.get_sta_templates_peeloff(emg_grid_test, pred_dts, R=R, L=50)
                    sep_mat_test = sta_test @ inv_cov_test
                    sources = (sep_mat_test @ extended_emg_test).T

                # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
                pred_dts, sils = utils.get_silohuette(sources)
                matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts2, pred_dts, old_matches=official_matches, fs=fsamp)
                fr_train, fr_pred = utils.get_mean_firing_rate(mu_dts, fs=fsamp), utils.get_mean_firing_rate(pred_dts, fs=fsamp)
                print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.8 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")
                print(f"Average firing rate difference between matches: {np.mean([np.abs(fr_train[key]-fr_pred[key]) for idx, key in enumerate(list(matches.keys())) if rate_of_agreement[idx] > 0.8])}")

                # Sort again based on SIL value
                sorted_idxs = np.argsort(sils)[::-1]
                mu_dts = [mu_dts[idx] for idx in sorted_idxs]
                pred_dts = [pred_dts[idx] for idx in sorted_idxs]
                _matches = dict(official_matches)
                official_matches = {idx: _matches[sort_idx] for idx, sort_idx in enumerate(sorted_idxs)}

            print(rate_of_agreement)

