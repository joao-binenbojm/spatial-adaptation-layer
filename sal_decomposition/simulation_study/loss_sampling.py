import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os

from sal_decomposition.simulation_study.sda_pipeline import SDAExperiment

mu_counts = [1, 10, 20] # number of MUs
fxmaxs = [62.5, 125, 187.5] # maximum spatial bandwidth in both grid directions

# Fixed simulation parameters
fs = 2000 # Hz
fsx = 250 # m^-1
duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 25, 10, 50
R = 16
sampfactor=15
SNR = 1
Tx, Ty = 1.5, -2.5 # chosen parameters for the translation
num_points = 100 # generate loss landscape
loss = 'kurtosis'
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on     

for mu_count in mu_counts:
    for fxmax in fxmaxs:
        exp = SDAExperiment()
        print('GENERATING MUAPS....')
        muaps = exp.generate_gaussian_muaps(mu_count, H, W, L, fxmax / (fsx/2), sampfactor) # generate MUAPs

        print('GENERATING SPIKE TRAINS...')
        spts, dts = exp.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains

        print('GENERATE EMG...')
        emg = exp.generate_emg(spts, muaps, R=R) # make synthetic EMG from simulated MUAPs and spike trains
        emg = exp.add_noise(emg, SNR) # add noise to synthetic signal
        emg = (emg - emg.mean(dim=2, keepdim=True)) / (emg.std(dim=2, keepdim=True) - 1e-9)
        emg_grid = exp.make_grid(emg) # reshape into EMG grid
        muaps = exp.downsample_muaps(muaps, sampfactor) # downsample MUAPs
        emg_grid_down = exp.downsample_grid(emg_grid, sampfactor) # downsample EMG grid

        print('GET SEPARATION VECTORS & WHITENING...')
        B = exp.get_separation_vectors(muaps, R=R)
        source_est = exp.get_whiten_mat(emg_grid_down, B, R=R)
        exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device) # get baseline loss


        with torch.no_grad():
            print('APPLY TRANSFORM AND DOWNSAMPLE...')
            emg_grid_transform = exp.apply_affine(emg_grid, Tx, Ty, 0.0, 1.0, 1.0, sampfactor)
            emg_grid_transform = exp.downsample_grid(emg_grid_transform, sampfactor)

            losses = exp.loss_sampling(emg_grid_transform.to(torch.float32), num_points=num_points, loss=loss, device=device)
            os.rename('./loss_landscapes/loss_landscape.jpg', f'./loss_landscapes/{mu_count}_{fxmax}.jpg')