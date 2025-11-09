import os
import sys

# Set the working directory to the directory of the script
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))  # Get the script's directory
os.chdir(script_dir)  # Change the current working directory to the script's directory

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from math import ceil

# from sal_decomposition.simulation_study._sda_pipeline import SDAExperiment
import sal_decomposition.utils.simulation_utils as sutils
from sal_decomposition.utils import utils
from sal_decomposition.sda import SpatialDecompositionAdaptation

mu_counts = [20] # number of MUs
fxmaxs = [62.5, 125, 187.5] # maximum spatial bandwidth in both grid directions

# Fixed simulation parameters
fs = 2000 # Hz
fsx = 250 # m^-1
duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 25, 10, 50
R = 16
sampfactor=15
mu_count = 20
SNR = 10
batch_size = 10000
delay = (torch.floor(torch.tensor([L + R])/2) - 1).to(torch.int) # delay introduced by causality of triggering process
reg = 5e-1

Tx, Ty = 1.5, -2.5 # chosen parameters for the translation
xcrop, ycrop = ceil(abs(Tx)), ceil(abs(Ty))
num_points = 20 # generate loss landscape
loss = 'negentropy'
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on     

with torch.no_grad():
    for fxmax in fxmaxs:
        print('GENERATING MUAPS....')
        muaps = sutils.generate_gaussian_muaps(mu_count, H, W, L, fxmax / (fsx/2), sampfactor) # generate MUAPs

        print('GENERATING SPIKE TRAINS...')
        spts, dts = sutils.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains

        print('GENERATE EMG...')
        emg = sutils.generate_emg(spts, muaps, device=device).to('cpu') # make synthetic EMG from simulated MUAPs and spike trains
        emg = sutils.add_noise(emg, SNR) # add noise to synthetic signal
        emg = (emg - emg.mean()) / (emg.std() + 1e-12) # standardize noisy emg
        emg_grid = sutils.simulation_make_grid(emg) # reshape into EMG grid
        emg_grid_down = sutils.downsample_grid(emg_grid, sampfactor) # downsample EMG grid
        
        # Get STA templates for separation vectors
        spts, dts = torch.roll(spts, (0, delay), dims=(0,1)), [dt + delay for dt in dts] # account for MUAP length delay
        spts[:, :delay] = 0.0 # remove any spikes that may have been rolled over the start of the signal
        muaps_down = sutils.downsample_muaps(muaps, sampfactor).to('cpu') # downsample MUAPs
        STA = sutils.get_sta_templates(muaps_down, R=R, delay=delay)        

        # Crop observations and get new sep_mat
        print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
        emg_grid_crop_train = emg_grid_down[:, :, ycrop:emg_grid_down.shape[2]-ycrop, xcrop:emg_grid_down.shape[3]-xcrop].clone()
        extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
        inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop_train, reg=reg).to(torch.float32)

        print('GET SEPARATION VECTORS & WHITENING...')
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R).to(device) # create SAL-Decomposition model
        base_loss = utils.get_base_loss(emg_grid_down, sda, batch_size=batch_size, loss=loss, device=device) # get baseline loss

        print('APPLY TRANSFORM AND DOWNSAMPLE...')
        emg_grid_transform = utils.apply_affine(emg_grid.detach().cpu().clone(), Tx*sampfactor, Ty*sampfactor, 0.0, mode='bicubic')
        emg_grid_transform = sutils.downsample_grid(emg_grid_transform, sampfactor).to(device)


        loss_arr = utils.loss_sampling(emg_grid_transform.clone(), sda.to(device), base_loss=base_loss, bounds=(2.5, 2.5), batch_size=batch_size, num_points=20, loss=loss, device=device)
        # os.rename('/home/joao/Desktop/spatial-adaptation-layer/sal_decomposition/simulation_study/loss_landscape.jpg',
        #             f'/home/joao/Desktop/spatial-adaptation-layer/loss_landscapes/{fxmax}.jpg')
        os.rename(r'C:\Users\Joao\Desktop\spatial-adaptation-layer\sal_decomposition\simulation_study\loss_landscape.jpg',
            rf'C:\Users\Joao\Desktop\spatial-adaptation-layer\sal_decomposition\simulation_study\loss_landscapes\loss_{fxmax}.jpg')