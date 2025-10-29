import numpy as np
from scipy import signal
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from math import floor

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2
import wandb


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

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    loss='negentropy'
    Nt = 50
    bounds = np.array([3.0, 3.0, 10*np.pi/180])
    sub_idx = 2
    
    R = 16
    reg = 1e-1
    
    DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'

    for mvc in [25, 50]:
        for ses_idx in [1,2]:
            file = f"S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat" 

            sgnl, edition = utils.open_mat_output(DIR, file)
            start, end = utils.get_target_boundaries(sgnl['target'].squeeze())

            # Apply filters to data and reshape into desired shape
            print('FILTER DATA...')
            emg = sgnl['data'][:, start:end]
            emg = emg - emg.mean(axis=0, keepdims=True) # average referencing
            emg = bandstop_filter(bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
            emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
            emg_grid = utils.make_grid(emg, index_matrix4).to(torch.float32)
            H, W = emg_grid.shape[2], emg_grid.shape[3]
            Nch = H*W                

            # Compute outliers as channels average of neighbours
            with open('./sal_decomposition/healthy_surface/outlier_channels.json', 'r') as f:
                outliers = json.load(f) 

            print('HANDLING OUTLIER CHANNELS...')
            visible_outliers = np.zeros((H, W), dtype=np.bool_)
            outlier_coords = outliers[f"subject{sub_idx+1}"][f"session{ses_idx+1}"]
            indices = tuple(np.array(outlier_coords).T)
            visible_outliers[indices] = True
            emg_grid = utils.handle_outliers(emg_grid, visible_outliers=visible_outliers)
            emg_grid = utils.handle_outliers(emg_grid)

            # Get separation matrix
            dts = edition['Dischargetimes']
            mu_dts = utils.squeeze_dts(dts)
            mu_dts = utils.filter_dts(mu_dts, start, end)

            # Loop over extension factor, explained variance, and transformations
            for trans_idx in range(Nt): # Nt random transformations
                Tx, Ty, theta = torch.tensor(np.random.uniform(-bounds, bounds)).to(torch.float32)
                print(f"Transformation parameters: Tx: {Tx}, Ty: {Ty}, rot_theta: {theta}")#, xscale: {xscale}, yscale: {yscale}")

                # Initialize wandb run
                run = wandb.init(
                    entity='jp2717-imperial-college-london',
                    project='real-data-simulations',
                    name=f'sub_{sub_idx+1}_ses_{ses_idx+1}_mvc_{mvc}',
                    config={'Subject': sub_idx+1, 'Session':ses_idx+1, 'MVC': mvc,
                            'Tx': Tx, 'Ty': Ty, 'theta': theta}, #, 'xscale': xscale, 'yscale': yscale},
                    mode='disabled'
                )

                # Get minimum distance between original and transformed grid
                original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
                print(f'MIN DISTANCE BETWEEN PIXELS: {min_distance} pixels')
                wandb.log({'min_distance': min_distance})

                # Test that masking channels is a valid solution
                print('TESTING MASKING CHANNELS...')
                extended_emg = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
                inv_cov = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
                STA = utils.get_sta_templates(extended_emg, mu_dts).to(torch.float32) # gets STA templates from Session 1 from the raw data

                sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov, extension_factor=R)
                base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss=loss, device='cpu')

                # Test on original grid with non-regularized whitening matrix
                with torch.no_grad():
                    source_est = sda(emg_grid)
                pred_dts, sils = utils.get_silohuette(source_est)
                matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                print('ROA Training:', np.mean(rate_of_agreement))
                print('#mu_train:', sum([roa > 0.7 for roa in rate_of_agreement]))
                wandb.log({'roa_train': np.mean(rate_of_agreement)})
                wandb.log({'roa_train_std': np.std(rate_of_agreement)})
                wandb.log({'#mu_train': sum([roa > 0.7 for roa in rate_of_agreement])})

                # Add optimal parameters for testing
                with torch.no_grad():
                    sda.sal.xshift[0].copy_(2*Tx/(W-1))
                    sda.sal.yshift[0].copy_(2*Ty/(H-1))
                    sda.sal.rot_theta[0].copy_(theta/np.pi)
                    emg_grid_test = sda.sal(emg_grid) # apply spatial transformation to get simulated data
                
                extended_emg = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
                sda.inv_cov = inv_cov_test

                with torch.no_grad():
                    source_est_valid = sda(emg_grid_test)
                pred_dts, sils = utils.get_silohuette(source_est_valid)
                matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                print('ROA Optimal:', np.mean(rate_of_agreement))
                print('#mu_opt_test:', sum([roa > 0.7 for roa in rate_of_agreement]))
                wandb.log({'roa_opt_test': np.mean(rate_of_agreement)})
                wandb.log({'roa_opt_test_std': np.std(rate_of_agreement)})
                wandb.log({'#mu_opt_test': sum([roa > 0.7 for roa in rate_of_agreement])})

                # Reset SDA-SAL parameters
                with torch.no_grad():
                    sda.sal.xshift[0].copy_(0.0)
                    sda.sal.yshift[0].copy_(0.0)
                    sda.sal.rot_theta[0].copy_(0.0)

                # Fit SDA model to determine optimal spatial transformation
                loss_arr = utils.loss_sampling(emg_grid_test.clone(), sda.to(device), base_loss=base_loss, bounds=bounds[:2], batch_size=batch_size, num_points=20, loss='negentropy', device=device)
                sources, losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=250, lr=5e-3, boundaries=bounds[:-2], device='cuda', loss=loss)

                # Log parameters
                Tx_est = (W-1)*sda.sal.xshift[0].item()/2
                Ty_est = (H-1)*sda.sal.yshift[0].item()/2
                theta_est = np.pi*sda.sal.rot_theta[0].item()
                wandb.log({'Tx_est': Tx_est, 'Ty_est': Ty_est, 'theta_est': theta_est})

                # Estimating the efficacy of spatial adaptation
                theta1 = utils.get_theta(emg_grid_test.shape, Tx=Tx, Ty=Ty, theta=theta)
                theta2 = utils.get_theta(emg_grid_test.shape, Tx=Tx_est, Ty=Ty_est, theta=theta_est)
                # theta_net = utils.get_theta(emg_grid_test.shape, Tx=Tx_est-Tx, Ty=Ty_est-Ty, theta=theta_est-theta)
                distance = utils.get_distance(emg_grid_test.shape, theta1, theta2) # get distance in pixels between initial and final location
                wandb.log({'transformation_distance': distance})
                print("Average Post-Corretion Distance (mm): ", 4*distance)


                pred_dts, sils = utils.get_silohuette(sources.to('cpu'))
                matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                print("ROA Test:", np.mean(rate_of_agreement))
                print('#mu_test: ', sum([roa > 0.7 for roa in rate_of_agreement]))
                wandb.log({'roa_test': np.mean(rate_of_agreement)})
                wandb.log({'roa_test_std': np.std(rate_of_agreement)})
                wandb.log({'#mu_test': sum([roa > 0.7 for roa in rate_of_agreement])})
                
                match_data = []
                for idx, (pred_idx, true_idx) in enumerate(matches.items()):
                    # Bundle everything together
                    match_info = {
                        "mu_dts": mu_dts[pred_idx],
                        "pred_dts": pred_dts[pred_idx],
                        "sil": sils[pred_idx],
                        "roa": rate_of_agreement[idx]
                    }
                    match_data.append(match_info)

                # Sort by SIL
                match_data = sorted(match_data, key=lambda x: x['roa'], reverse=True)


                # Get new covariance matrix 
                extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg_test, reg=1e-3) #.to(torch.float32)

                # Refinement of MUs detected
                Nr = 20
                # official_matches = dict(matches)
                for idx in tqdm(range(Nr)):
                    # Get sorted pred_dts
                    pred_dts = [item['pred_dts'] for item in match_data]

                    with torch.no_grad():
                        sta_test = utils.get_sta_templates_peeloff(emg_grid_test.clone(), pred_dts, R=R, L=50)
                        sep_mat_test = sta_test @ inv_cov_test
                        sources = (sep_mat_test @ extended_emg_test).T

                    # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
                    pred_dts, sils = utils.get_silohuette(sources)
                    for i, item in enumerate(match_data):
                        item['pred_dts'] = pred_dts[i]
                        item['sil'] = sils[i]

                    mu_dts = [item['mu_dts'] for item in match_data]

                    # Compute metrics
                    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, old_matches={idx:idx for idx in range(len(pred_dts))}, fs=fsamp)
                    print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.7 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")

                    # Adding RoA to match data
                    for i, item in enumerate(match_data):
                        item['roa'] = rate_of_agreement[i]

                    # Sort again based on SIL value
                    match_data = sorted(match_data, key=lambda x: x['roa'], reverse=True)

                print(rate_of_agreement)
                print()                
                
                wandb.log({'roa_refine': np.mean(rate_of_agreement)})
                wandb.log({'roa_refine_std': np.std(rate_of_agreement)})
                wandb.log({'#mu_refine': sum([roa > 0.7 for roa in rate_of_agreement])})
                wandb.finish()