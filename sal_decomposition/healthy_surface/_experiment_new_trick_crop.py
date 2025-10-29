import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from math import floor

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptationOld, SpatialDecompositionAdaptation
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

# In the order of the cables, it is -----> OLD AND WRONG VERSION
# GRID 4    GRID 3
# GRID 1    GRID 2
# So taking the 256 signals in signal.data as input, one must reshape in
# the following way:

# index_matrix4[13:26,5:10] =  index_matrix4[13:26,5:10] + 64 
# index_matrix4[0:13,5:10] = index_matrix4[0:13,5:10] + 64 + 64 
# index_matrix4[0:13,0:5] = index_matrix4[0:13,0:5] + 64 + 64 + 64

index_matrix4[0:13, 5:10] =  index_matrix4[0:13,5:10] + 64
index_matrix4[13:26, 5:10] = index_matrix4[13:26,5:10] + 64 + 64
index_matrix4[13:26, 0:5] = index_matrix4[13:26,0:5] + 64 + 64 + 64


# # Matrix reshaping!
# index_matrix4 = np.array([
#     [-1, 63, 62, 64, 52, 49, 47, 40, 48, -1],
#     [53, 55, 54, 50, 51, 52, 42, 41, 33, 43],
#     [56, 61, 42, 41, 49, 53, 50, 51, 44, 34],
#     [57, 58, 45, 44, 43, 54, 55, 39, 45, 46],
#     [59, 60, 48, 47, 46, 56, 64, 35, 36, 38],
#     [33, 34, 38, 39, 40, 63, 62, 26, 37, 27],
#     [2, 1, 25, 36, 37, 61, 60, 28, 29, 31],
#     [3, 15, 27, 26, 35, 59, 58, 19, 30, 32],
#     [4, 24, 32, 31, 28, 57, 1, 4, 20, 21],
#     [16, 21, 22, 23, 29, 2, 3, 16, 22, 23],
#     [8, 18, 19, 20, 30, 5, 6, 11, 24, 25],
#     [6, 7, 10, 9, 17, 7, 15, 12, 17, 18],
#     [14, 5, 13, 12, 11, 8, 14, 13, 9, 10],
#     [62, 53, 52, 51, 50, 54, 50, 42, 40, 43],
#     [54, 49, 42, 43, 44, 64, 53, 52, 45, 44],
#     [55, 41, 45, 46, 47, 62, 63, 51, 47, 46],
#     [56, 64, 48, 40, 39, 60, 61, 55, 41, 48],
#     [63, 57, 38, 37, 36, 57, 58, 56, 49, 35],
#     [58, 59, 35, 30, 31, 2, 1, 59, 37, 36],
#     [60, 61, 29, 32, 24, 4, 3, 34, 39, 38],
#     [34, 33, 25, 23, 22, 6, 5, 26, 25, 33],
#     [26, 27, 28, 21, 20, 8, 7, 29, 28, 27],
#     [1, 2, 3, 19, 18, 16, 15, 14, 11, 30],
#     [4, 16, 15, 17, 9, 12, 13, 31, 32, 23],
#     [14, 5, 6, 11, 10, 17, 19, 22, 21, 24],             # B AT BOTTOM RIGHT HAD ORIGINALLY TWO 19s and no 9. I SWAPPED THE BOTTOM 19 FOR THE 9
#     [-1, 7, 8, 13, 12, 10, 9, 18, 20, -1]
# ])
# index_matrix4 = index_matrix4 - 1

# In the order of the cables, it is -----> OLD AND WRONG VERSION
# GRID C    GRID D  
# GRID A    GRID B

# BECOMES
# GRID 1    GRID 2
# GRID 4    GRID 3

# index_matrix4[0:13, 5:10] =  index_matrix4[0:13,5:10] + 64
# index_matrix4[13:26, 5:10] = index_matrix4[13:26,5:10] + 64 + 64
# index_matrix4[13:26, 0:5] = index_matrix4[13:26,0:5] + 64 + 64 + 64


# # 2MM GRID TRYING SET-UP
# index_matrix4 = np.array([
#     [-1, 55, 53, 62, 63, 48, 40, 43, 44, -1],
#     [61, 56, 54, 52, 64, 47, 49, 33, 34, 45],
#     [57, 43, 49, 50, 51, 41, 42, 52, 53, 46],
#     [60, 58, 44, 42, 41, 51, 50, 54, 35, 36],
#     [34, 59, 47, 46, 45, 55, 56, 64, 26, 37],
#     [25, 33, 39, 40, 48, 63, 62, 61, 38, 27],
#     [27, 26, 36, 37, 38, 60, 59, 58, 39, 28],
#     [1, 28, 30, 29, 35, 57, 1, 2, 29, 19],
#     [3, 2, 24, 32, 31, 3, 4, 5, 30, 31],
#     [16, 4, 21, 22, 23, 6, 7, 8, 32, 20],
#     [7, 8, 18, 19, 20, 16, 15, 14, 21, 22],
#     [5, 6, 10, 9, 17, 11, 12, 13, 23, 24],
#     [14, 15, 13, 12, 11, 17, 9, 10, 25, 18],
#     [62, 53, 52, 51, 50, 42, 41, 49, 43, 40],
#     [54, 55, 49, 41, 42, 50, 51, 52, 45, 44],
#     [56, 64, 43, 44, 45, 55, 54, 53, 47, 46],
#     [63, 57, 46, 47, 48, 63, 64, 56, 35, 48],
#     [58, 59, 40, 39, 38, 60, 61, 62, 37, 36],
#     [60, 61, 37, 36, 35, 57, 58, 59, 39, 38],
#     [34, 33, 29, 30, 31, 3, 2, 1, 33, 34],
#     [25, 26, 32, 24, 23, 6, 5, 4, 29, 28],
#     [27, 28, 22, 21, 20, 16, 8, 7, 31, 30],
#     [1, 9, 17, 10, 12, 12, 11, 15, 23, 32],
#     [2, 18, 19, 11, 13, 9, 17, 13, 14, 22],  # in first connector (A), there are two 19s and no 9, replace top 19 with 9 and see what happens
#     [3, 4, 7, 5, 14, 10, 19, 26, 24, 21],
#     [-1, 6, 8, 15, 16, 20, 18, 27, 25, -1]
# ])
# index_matrix4 = index_matrix4 - 1

# In the order of the cables, it is -----> OLD AND WRONG VERSION
# GRID 3    GRID 4
# GRID 1    GRID 2
# BECOMES
# GRID 1    GRID 2
# GRID 4    GRID 3

# index_matrix4[13:26, 5:10] =  index_matrix4[13:26,5:10] + 64
# index_matrix4[0:13, 0:5] = index_matrix4[0:13,0:5] + 64 + 64
# index_matrix4[0:13, 5:10] = index_matrix4[0:13,5:10] + 64 + 64 + 64

# index_matrix4[0:13, 5:10] =  index_matrix4[0:13,5:10] + 64
# index_matrix4[13:26, 5:10] = index_matrix4[13:26,5:10] + 64 + 64
# index_matrix4[13:26, 0:5] = index_matrix4[13:26,0:5] + 64 + 64 + 64


## FROM OLD MAPPING FILE OF 4MM
# index_matrix4 = np.array([[53,56,57,59,33,2,3,4,16,8,6,14,254,246,247,248,255,250,252,226,218,193,196,206],
#         [55,61,58,60,34,1,15,24,21,18,7,5,245,241,233,256,249,251,253,225,219,194,208,197],
#         [54,42,45,48,38,25,27,32,22,19,10,13,244,234,237,240,230,227,221,217,220,195,207,198],
#         [50,41,44,47,39,36,26,31,23,20,9,12,243,235,238,232,229,222,224,215,213,211,209,203],
#         [51,49,43,46,40,37,35,28,29,30,17,11,242,236,239,231,228,223,216,214,212,210,201,202],
#         [116,117,118,120,127,125,123,121,66,69,71,72,182,192,190,188,185,130,132,134,136,144,140,145],
#         [106,114,119,128,126,124,122,65,67,70,79,78,178,181,191,189,186,129,131,133,135,143,141,147],
#         [105,115,103,99,90,92,83,68,80,75,76,77,170,180,179,183,184,187,162,154,157,142,159,150],
#         [97,108,109,100,101,93,94,84,86,88,81,73,168,173,175,169,177,165,167,153,156,139,160,149],
#         [107,98,110,102,91,95,96,85,87,89,82,74,171,172,174,176,163,164,166,161,155,158,151,152]]).T - 1



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    Tx_max, Ty_max, theta_max  = 3, 3, 0*np.pi/180

    for sub_idx in [1]:#range(2):
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'
        if sub_idx == 0: DIR = os.path.join(DIR, '4mm')

        for mvc in [25, 50]:
            for ses_idx in range(3):
                file = f'S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat'

                signal, edition = utils.open_mat_output(DIR, file)
                start, end = utils.get_target_boundaries(signal['target'].squeeze())
                # torch.set_default_dtype(torch.float64)

                # Apply filters to data and reshape into desired shape
                print('FILTER DATA...')
                emg = signal['data'][:, start:end]
                emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
                emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)
                emg_grid = utils.make_grid(emg, index_matrix4).to(torch.float32)
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
                # xcrop, ycrop = 0, 0

                # Loop over extension factor, explained variance, and transformations
                for R in [16]:#[8, 16, 32]:
                    for explained_var in [1-1e-2, 1-1e-3, 1-1e-4, 1-1e-6]:
                        explained_var = 1-1e-4
                        for trans_idx in range(30): # thirty random transformations
                            params = np.random.uniform([-Tx_max, -Ty_max, -theta_max], [Tx_max, Ty_max, theta_max]) # sample random transformation parameters
                            Tx, Ty, theta = torch.tensor(params).to(torch.float32)    

                            # Initialize wandb run
                            run = wandb.init(
                                entity='jp2717-imperial-college-london',
                                project='real-data-simulations',
                                name=f'sub_{sub_idx+1}_ses_{ses_idx+1}_mvc_{mvc}',
                                config={'Subject': sub_idx+1, 'Session':ses_idx+1, 'MVC': mvc,
                                        'Extension Factor': R, 'Explained Variance': explained_var, 
                                        'Tx': Tx, 'Ty': Ty, 'theta': theta, '#mu':len(mu_dts)},
                                mode='disabled'
                            )

                            # Crop observations and get new sep_mat
                            print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
                            emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()

                            # Get minimum distance between original and transformed grid
                            original_grid, transformed_grid, min_distance = utils.get_min_distance((H, W), Tx, Ty, theta)
                            print(f'MIN DISTANCE: {min_distance} pixels')
                            wandb.log({'min_distance': min_distance})

                            # Create mask based on transformed coordinates being within convex 
                            lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, original_grid)

                            # Test that masking channels is a valid solution
                            print('TESTING MASKING CHANNELS...')
                            # emg_grid_valid = emg_grid[:, :, tcrop:H-bcrop, lcrop:W-rcrop]
                            extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
                            extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
                            # inv_cov_valid = utils.get_inv_cov_torch(extended_emg_crop_train, explained_var=explained_var)
                            inv_cov_crop_train = utils.get_inv_cov_torch(extended_emg_crop_train, explained_var=explained_var).to(torch.float32)
                            STA = utils.get_sta_templates(extended_emg_train, mu_dts).to(torch.float32)
                            # sep_mat_crop_train = STA @ inv_cov_crop_train
                            # sep_mat_valid_train = STA @ inv_cov_valid_train

                            # sda = SpatialDecompositionAdaptationOld(grid_shape=(H, W), sep_mat=sep_mat_crop_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
                            sda = SpatialDecompositionAdaptation(grid_shape=(H,W), STA=STA, inv_cov=inv_cov_crop_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)  
                            # sda.lcrop, sda.rcrop = lcrop, rcrop
                            # sda.bcrop, sda.tcrop = bcrop, tcrop
                            sda.sal.mode = 'bicubic'

                            # Test on original grid with non-regularized whitening matrix
                            with torch.no_grad():
                                source_est_valid = sda(emg_grid)
                            pred_dts, sils = utils.get_silohuette(source_est_valid)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print('F1 Scores Training:', np.mean(f1_scores))
                            print('#mu_train:', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_train': np.mean(f1_scores)})
                            wandb.log({'#mu_train': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Add optimal parameters for testing
                            # sda.sep_mat.weight = torch.nn.Parameter(sep_mat_valid)
                            with torch.no_grad():
                                sda.sal.xshift[0].copy_(2*Tx/(W-1))
                                sda.sal.yshift[0].copy_(2*Ty/(H-1))
                                sda.sal.rot_theta[0].copy_(theta/np.pi)

                                # Apply affine transform to full grid to get test grid
                                emg_grid_test = sda.apply_affine(emg_grid)
                                emg_grid_test_crop = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
                                extended_emg_crop_test = utils.extend_emg_torch(emg_grid_test_crop.squeeze().reshape(emg_grid_test_crop.shape[0], -1), R).T
                                inv_cov_crop_test = utils.get_inv_cov_torch(extended_emg_crop_test, explained_var=explained_var).to(torch.float32)
                                sda.inv_cov = inv_cov_crop_test

                            # Test on optimal forward
                            with torch.no_grad():
                                source_est_valid = sda(emg_grid_test)

                            pred_dts, sils = utils.get_silohuette(source_est_valid)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print('F1 Scores Optimal:', np.mean(f1_scores))
                            print('#mu_opt_test:', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_opt_test': np.mean(f1_scores)})
                            wandb.log({'#mu_opt_test': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Reset SDA-SAL parameters
                            with torch.no_grad():
                                sda.sal.xshift[0].copy_(0.0)
                                sda.sal.yshift[0].copy_(0.0)
                                sda.sal.rot_theta[0].copy_(0.0)

                            base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss='kurtosis', device='cpu')

                            sda.sal.mode = 'bilinear'
                            sources, losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=1000, nepochs=50, lr=1e-3, boundaries=(Tx_max, Ty_max, theta_max), device='cuda')

                            # Log parameters
                            wandb.log({'Tx_est':(W-1)*sda.sal.xshift[0].item()/2, 'Ty_est':(H-1)*sda.sal.yshift[0].item()/2, 'theta_est':np.pi*sda.sal.rot_theta[0].item()})

                            # Update sda for testing
                            sda = sda.to('cpu')
                            sda.sal.mode = 'bicubic'

                            # Obtain new separation matrix with all valid channels
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
                            pred_dts, sils = utils.get_silohuette(sources)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            print('#mu_test: ', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_test': np.mean(f1_scores)})
                            wandb.log({'#mu_test': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Get sep mat based on real test data
                            print('Getting separation matrix based on real test data...')
                            sda.lcrop, sda.rcrop, sda.bcrop, sda.tcrop = 0, 0, 0, 0
                            extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).to(torch.float32).T
                            inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=explained_var).to(torch.float32)
                            sta_test = utils.get_sta_templates(extended_emg_test, pred_dts).to(torch.float32)
                            sep_mat_test = sta_test @ inv_cov_test
                            with torch.no_grad():
                                sources = (sep_mat_test @ extended_emg_test).T

                            pred_dts, sils = utils.get_silohuette(sources)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            print('#mu_refine:', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_refine': np.mean(f1_scores)})
                            wandb.log({'#mu_refine': sum([f1_score > 0.8 for f1_score in f1_scores])})
                            wandb.finish()