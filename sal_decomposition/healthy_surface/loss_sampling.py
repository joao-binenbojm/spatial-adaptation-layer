import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from math import ceil
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2
from sal_decomposition.utils import utils
from sal_decomposition.sda import SpatialDecompositionAdaptation

# Model parameters
fsamp = 2048 # Hz
fsx = 250 # m^-1
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 26, 10, 50
SNR = 10
batch_size = 10000
reg = 5e-1
R = 16

num_points = 20 # generate loss landscape
loss = 'negentropy'
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on     

with torch.no_grad():
    
    for sub_idx in [0,1,2]: 
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'
        if sub_idx == 0:
            DIR = os.path.join(DIR, '4mm')

        for mvc in [25, 50]:
            for ses_idx in [0,1,2]:
                
                file = f"S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat" 
                sgnl, edition = utils.open_mat_output(DIR, file)
                start, end = utils.get_target_boundaries(sgnl['target'].squeeze())

                # Define cropping parameters
                Tx, Ty = np.random.uniform(-np.array([2.5, 2.5]), np.array([2.5, 2.5]))
                xcrop, ycrop = 3, 5

                # Apply filters to data and reshape into desired shape
                print('FILTER DATA...')
                emg = sgnl['data'][:, start:end]
                emg = emg - emg.mean(axis=0, keepdims=True) # average referencing
                emg = utils.bandstop_filter(utils.bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
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

                # Get separation matrix
                dts = edition['Dischargetimes']
                mu_dts = utils.squeeze_dts(dts)
                mu_dts = utils.filter_dts(mu_dts, start, end)

                # Get base metrics
                print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
                emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
                extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
                inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop_train, reg=reg).to(torch.float32)

                # Test that masking channels is a valid solution
                print('TESTING MASKING CHANNELS...')
                extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
                STA = utils.get_sta_templates(extended_emg_train, mu_dts).to(torch.float32) # gets STA templates from Session 1 from the raw data

                # Loop over extension factor, explained variance, and transformations
                print(f"S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat")

                sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
                base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss=loss, device='cpu')

                # Add optimal parameters for testing
                with torch.no_grad():
                    sda.sal.xshift[0].copy_(2*Tx/(W-1))
                    sda.sal.yshift[0].copy_(2*Ty/(H-1))
                    emg_grid_test = sda.sal(emg_grid) # apply spatial transformation to get simulated data

                # Update inverse covariance matrix based on session 2 data
                emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
                extended_emg = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
                inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
                sda.inv_cov = inv_cov_test

                # Resetting spatial parameters
                with torch.no_grad():
                    sda.sal.xshift[0].copy_(0.0)
                    sda.sal.yshift[0].copy_(0.0)
                    sda.sal.rot_theta[0].copy_(0.0)

                loss_arr = utils.loss_sampling(emg_grid_test.clone(), sda.to(device), base_loss=base_loss, bounds=(2.5, 2.5), batch_size=batch_size, num_points=20, loss=loss, device=device)
                # os.rename('/home/joao/Desktop/spatial_adaptation_layer/sal_decomposition/healthy_surface/loss_landscape.jpg',
                os.rename('/home/joao/Desktop/spatial_adaptation_layer/loss_landscape.jpg',
                    f'/home/joao/Desktop/spatial_adaptation_layer/sal_decomposition/healthy_surface/loss_landscapes/{sub_idx+1}-{ses_idx+1}-{mvc}.jpg')
                