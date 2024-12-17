import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import wandb
import pandas as pd

from sal_decomposition.simulation_study.sda_pipeline import SDAExperiment

# Define experimental checklist to include all conditions already tried and ran
# this will allow us to continue where we left off if the system breaks
# checklist is a set of tuples of experimental conditions

checklist = []

# Define experimental parameters
mu_counts = [1, 5, 10, 15, 20]
SNRs = [30, 15, 5, 1]
fxmaxs = [62.5, 93.75, 125, 156.25, 187.5] # m^-1
opts = ['fit', 'search', 'search_fit'] # whether to only train, only search, or search and fit

# Fixed simulation parameters
fs = 2000 # Hz
fsx = 250 # m^-1
duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 25, 10, 50
R = 16
sampfactor=15

# Training params
nepochs=100
lr = 5e-3
loss = 'kurtosis' # loss function for optimization
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

for opt in opts:
    for mu_count in tqdm(mu_counts):
        for SNR in SNRs:
            for fxmax in fxmaxs:
                
                # If in checklist, already run, continue to next condition
                if (opt, mu_count, SNR, fxmax) in checklist:
                    continue

                with torch.no_grad():
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

                    # Get baseline performance metrics before transform (mainly to evaluate initial decomp.)

                # Test 30 randomly sampled spatial transformations
                for trans_idx in range(30):
                    Tx, Ty = np.random.uniform(-3.0, 3.0), np.random.uniform(-3.0, 3.0)
                    theta = np.random.uniform(-20*np.pi/180, 20*np.pi/180)
                    xscale, yscale = np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)

                    # Start the wandb run
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="sda-simulations",
                        name=f'{opt}-{mu_count}-{SNR}-{fxmax}',
                        # mode='disabled',
                    )
                    
                    # Keep track of experimental parameters of the run
                    params = {'opt': opt, 'mu_count': mu_count, 'SNR':SNR, 'fxmax': fxmax,
                                'Tx': Tx, 'Ty': Ty, 'theta': theta, 'xscale': xscale, 'yscale': yscale}

                    with torch.no_grad():
                        print('APPLY TRANSFORM...')
                        emg_grid_transform = exp.apply_affine(emg_grid, Tx, Ty, theta, xscale, yscale, sampfactor)

                        print('DOWNSAMPLING...')
                        emg_grid_transform = exp.downsample_grid(emg_grid_transform, sampfactor)

                    # Optimization
                    if opt == 'fit':
                        sources, losses = exp.search_fit_sda(emg_grid_transform.to(torch.float32), n_points=0, nepochs=nepochs, lr=lr, device=device, loss=loss, plot=0)
                    elif opt == 'search_fit':
                        sources, losses = exp.search_fit_sda(emg_grid_transform.to(torch.float32), npoints=3*nepochs//2, nepochs=nepochs//2, lr=lr, device=device, loss=loss, plot=0)
                    else: # search only
                        sources, losses = exp.search_fit_sda(emg_grid_transform.to(torch.float32), npoints=3*nepochs, nepochs=0, lr=lr, device=device, loss=loss, plot=0)
                    
                    # Get learned transformations
                    Tx_opt, Ty_opt = W*exp.sda.sal.xshift.item()/2, H*exp.sda.sal.yshift.item()/2
                    theta_opt = np.pi*exp.sda.sal.rot_theta.item()
                    xscale_opt, yscale_opt = exp.sda.sal.xscale.item(), exp.sda.sal.yscale.item()

                    params.update({'Tx_opt': Tx_opt, 'Ty_opt':Ty_opt, 'theta_opt': theta_opt,
                                   'xscale_opt':xscale_opt, 'yscale_opt': yscale_opt})

                    # Performance metrics based on output sources
                    pred_dts, sils = exp.get_silohuette(sources.detach().cpu().numpy())
                    scores = exp.spike_scores(dts, pred_dts)
                    print('SILS:', sils)
                    print()
                    print('SCORES:', scores)

                    params.update({'sils_avg': np.mean(sils), 'sils_std': np.std(sils),
                                   'sensitivity_avg': np.mean(scores['sensitivity']), 'sensitivity_std': np.std(scores['sensitivity']),
                                   'precision_avg': np.mean(scores['precision']), 'precision_std': np.std(scores['precision'])})

                    # Finish wandb run with all scores and parameters of the system
                    wandb.log(params)
                # Add parameters to checklist besides the affine transformation ones
                checklist.append((opt, mu_count, SNR, fxmax))

