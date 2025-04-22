import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from math import floor

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
import wandb
                 

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

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    Tx_max, Ty_max, theta_max  = 2, 2, 10*np.pi/180

    for sub_idx in range(2):
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'
        if sub_idx == 0: DIR = os.path.join(DIR, '4mm')

        for mvc in [25, 50]:
            for ses_idx in range(3):
                file = f'S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat'

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

                # Get conservative crop
                delta_width, delta_height = utils.out_of_bounds_pixels(H, W, theta_max)
                xcrop, ycrop = Tx_max + floor(delta_width + 0.5), Ty_max + floor(delta_height + 0.5)

                # Loop over extension factor, explained variance, and transformations
                for R in [8, 16, 32]:
                    for explained_var in [1-1e-2, 1-1e-3, 1-1e-4, 1-1e-6]:
                        for trans_idx in range(30): # thirty random transformations
                            Tx, Ty, theta = np.random.uniform([-Tx_max, -Ty_max, -theta_max], [Tx_max, Ty_max, theta_max]) # sample random transformation parameters

                            # Initialize wandb run
                            run = wandb.init(
                                entity='jp2717-imperial-college-london',
                                project='real-data-simulations',
                                name=f'sub_{sub_idx+1}_ses_{ses_idx+1}_mvc_{mvc}',
                                config={'Subject': sub_idx+1, 'Session':ses_idx+1, 'MVC': mvc,
                                        'Extension Factor': R, 'Explained Variance': explained_var, 
                                        'Tx': Tx, 'Ty': Ty, 'theta': theta, '#mu':len(mu_dts)}
                            )


                            # Crop observations and get new sep_mat
                            print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
                            emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
                            emg_grid_test = utils.apply_affine(emg_grid, Tx, Ty, theta, mode='bicubic') #theta) # Apply affine transformation and appropriate padding

                            # Get minimum distance between original and transformed grid
                            original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
                            print(f'MIN DISTANCE: {min_distance} pixels')
                            wandb.log({'min_distance': min_distance})

                            # Create mask based on transformed coordinates being within convex 
                            lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

                            # Test that masking channels is a valid solution
                            print('TESTING MASKING CHANNELS...')
                            emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
                            extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
                            inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var)
                            inv_cov_valid_train = utils.get_inv_cov_torch(extended_emg_valid, explained_var=1-1e-12)
                            sep_mat = utils.get_sep_mat_torch(extended_emg_valid, mu_dts)
                            sep_mat_valid = sep_mat @ inv_cov_valid
                            sep_mat_valid_train = sep_mat @ inv_cov_valid_train

                            sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=sep_mat_valid_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
                            sda.lcrop, sda.rcrop = lcrop, rcrop
                            sda.bcrop, sda.tcrop = bcrop, tcrop
                            sda.sal.mode = 'bicubic'

                            # Test on original grid with non-regularized whitening matrix
                            with torch.no_grad():
                                source_est_valid = sda(emg_grid)
                            pred_dts, sils = utils.get_silohuette(source_est_valid)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print('F1 Scores Training:', np.mean(f1_scores))
                            wandb.log({'f1_train': np.mean(f1_scores)})
                            wandb.log({'#mu_train': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Add optimal parameters for testing
                            sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
                            with torch.no_grad():
                                sda.sal.xshift.copy_(-2*Tx/W)
                                sda.sal.yshift.copy_(-2*Ty/H)
                                sda.sal.rot_theta.copy_(-theta/np.pi)

                            # Test on optimal inverse transformation
                            with torch.no_grad():
                                source_est_valid = sda(emg_grid_test)
                            pred_dts, sils = utils.get_silohuette(source_est_valid)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print('F1 Scores Optimal:', np.mean(f1_scores))
                            wandb.log({'f1_opt_test': np.mean(f1_scores)})
                            wandb.log({'#mu_opt_test': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Reset to cropped separation matrix and conservative cropping
                            sda.lcrop, sda.rcrop = xcrop, xcrop
                            sda.bcrop, sda.tcrop = ycrop, ycrop
                            extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
                            # inv_cov = get_inv_cov_tikhonov(extended_emg_crop_train, reg=tikhonov)
                            inv_cov = utils.get_inv_cov_torch(extended_emg_crop_train, explained_var=explained_var)
                            sep_mat_crop_train = utils.get_sep_mat_torch(extended_emg_crop_train, mu_dts)
                            sep_mat_crop_train = sep_mat_crop_train @ inv_cov
                            sda.sep_mat.weight = torch.nn.Parameter(sep_mat_crop_train)

                            # Reset SDA-SAL parameters
                            with torch.no_grad():
                                sda.sal.xshift.copy_(0.0)
                                sda.sal.yshift.copy_(0.0)
                                sda.sal.rot_theta.copy_(0.0)

                            base_loss = utils.get_base_loss(emg_grid.to(torch.float64), sda, batch_size=batch_size, loss='kurtosis', device='cpu')

                            sda.sal.mode = 'bilinear'
                            losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=500, nepochs=50, lr=1e-3, boundaries=(Tx_max, Ty_max, theta_max), device='cuda')

                            # Log parameters
                            wandb.log({'Tx_est':sda.sal.xshift.item(), 'Ty_est':sda.sal.yshift.item(), 'theta_est':sda.sal.rot_theta.item()})

                            # Update sda for testing
                            sda = sda.to('cpu')
                            sda.sal.mode = 'bicubic'

                            # Obtain new separation matrix with all valid channels
                            print('Obtaining new separation matrix...')
                            sda.lcrop, sda.rcrop = lcrop, rcrop
                            sda.bcrop, sda.tcrop = bcrop, tcrop
                            emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
                            extended_emg_valid = utils.extend_emg_torch(emg_grid_valid.squeeze().reshape(emg_grid_valid.shape[0], -1), R).T
                            # inv_cov_valid = get_inv_cov_tikhonov(extended_emg_valid, reg=tikhonov)
                            inv_cov_valid = utils.get_inv_cov_torch(extended_emg_valid, explained_var=explained_var)
                            sep_mat_valid = utils.get_sep_mat_torch(extended_emg_valid, mu_dts)
                            sep_mat_valid = sep_mat_valid @ inv_cov_valid
                            sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
    
                            # Get initial source estimates
                            with torch.no_grad():
                                sources = sda(emg_grid_test)
                            pred_dts, sils = utils.get_silohuette(sources)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            wandb.log({'f1_test': np.mean(f1_scores)})
                            wandb.log({'#mu_test': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Get sep mat based on real test data
                            print('Getting separation matrix based on real test data...')
                            sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
                            if R == '1000/ch':
                                R = 1000/(emg_grid_test.shape[2]*emg_grid_test.shape[3])
                            extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                            inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=1-1e-12)
                            sep_mat_test = utils.get_sep_mat_torch(extended_emg_test, pred_dts)
                            sep_mat_test = sep_mat_test @ inv_cov_test
                            with torch.no_grad():
                                sources = (sep_mat_test @ extended_emg_test).T

                            pred_dts, sils = utils.get_silohuette(sources)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            wandb.log({'f1_refine': np.mean(f1_scores)})
                            wandb.log({'#mu_refine': sum([f1_score > 0.8 for f1_score in f1_scores])})
                            wandb.finish()