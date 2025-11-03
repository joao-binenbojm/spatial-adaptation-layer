import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import pickle
from math import floor
import wandb
import pandas as pd

import sal_decomposition.utils.simulation_utils as sutils
from sal_decomposition.utils import utils
from sal_decomposition.sda import SpatialDecompositionAdaptation

from sal_decomposition.simulation_study._sda_pipeline import SDAExperiment

# Define experimental checklist to include all conditions already tried and ran
# this will allow us to continue where we left off if the system breaks
# checklist is a set of tuples of experimental conditions

# # Get checklist of all runs done so far
checklist = []
# api = wandb.Api()
# runs = api.runs(f"jp2717-imperial-college-london/sal-decomposition-simulations3")
# checklist = []
# for run in runs:
#     try:
#         snr = run.summary.get("SNR")
#         fxmax = run.summary.get("fxmax")
#         opt = run.summary.get("opt")
#         if snr is not None and fxmax is not None and opt is not None:
#             checklist.append((fxmax, snr, opt))
#     except KeyError:
#         print(f"Skipping run {run.id} due to missing entries.")

# Define experimental parameters
mu_count = 20
exp_name = sys.argv[1]  # First argument after script name

with open(f"sal_decomposition/simulation_study/{exp_name}.json", "r") as f:
    exp = json.load(f)

SNR, opt, fxmax = exp['SNR'], exp['opt'], exp['fxmax']
print('EXPERIMENTAL CONDITIONS:', exp)

# Fixed simulation parameters
fsamp = 2000 # Hz
fsx = 250 # m^-1
duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 26, 10, 50
R = 16
sampfactor = 14
reg = 5e-1
delay = (torch.floor(torch.tensor([L + R])/2) - 1).to(torch.int) # delay introduced by causality of triggering process

# Training params
batch_size = 10000
nepochs = 250
lr = 5e-3
loss = 'negentropy' # loss function for optimization
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

# Create range of spatial transformations to be used equally for every single condition
bounds = [2.5, 2.5, 15*np.pi/180]
y_margin, x_margin = utils.out_of_bounds_pixels(26, 10, bounds[2])
xcrop, ycrop = int(bounds[0] + floor(x_margin + 0.5)), int(bounds[1] + floor(y_margin + 0.5))
Nt = 50
Txs, Tys = np.random.uniform(-bounds[0], bounds[0], size=Nt), np.random.uniform(-bounds[1], bounds[1], size=Nt)
thetas = np.random.uniform(-bounds[2], bounds[2], size=Nt)

with torch.no_grad():
    # exp = SDAExperiment()
    print('GENERATING MUAPS....')
    # muaps = exp.generate_gaussian_muaps(mu_count, H, W, L, fxmax / (fsx/2), sampfactor) # generate MUAPs
    muaps = sutils.generate_gaussian_muaps(mu_count, H, W, L, fxmax / (fsx/2), sampfactor) # generate MUAPs
    print('GENERATING SPIKE TRAINS...')
    # spts, dts = exp.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains
    spts, dts = sutils.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains
    print('GENERATE EMG...')
    emg = sutils.generate_emg(spts, muaps, device=device).to('cpu') # make synthetic EMG from simulated MUAPs and spike trains
    spts, dts = torch.roll(spts, (0, delay), dims=(0,1)), [dt + delay for dt in dts] # account for MUAP length delay
    spts[:, :delay] = 0.0 # remove any spikes that may have been rolled over the start of the signal
    
    # Get STA templates for separation vectors
    muaps_down = sutils.downsample_muaps(muaps, sampfactor).to('cpu') # downsample MUAPs
    STA = sutils.get_sta_templates(muaps_down, R=R, delay=delay)
    
    # Add noise, make grid and downsample EMG
    noisy_emg = sutils.add_noise(emg, SNR) # add noise to synthetic signal
    noisy_emg = (noisy_emg - noisy_emg.mean()) / (noisy_emg.std() + 1e-12) # standardize noisy emg
    emg_grid = sutils.simulation_make_grid(noisy_emg) # reshape into EMG grid
    emg_grid_down = sutils.downsample_grid(emg_grid, sampfactor) # downsample EMG grid

    # Crop observations and get new sep_mat
    print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
    emg_grid_crop_train = emg_grid_down[:, :, ycrop:emg_grid_down.shape[2]-ycrop, xcrop:emg_grid_down.shape[3]-xcrop].clone()
    extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
    inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop_train, reg=reg).to(torch.float32)

    print('GET SEPARATION VECTORS & WHITENING...')
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R).to(device) # create SAL-Decomposition model
    base_loss = utils.get_base_loss(emg_grid_down, sda, batch_size=batch_size, loss=loss, device=device) # get baseline loss

    with torch.no_grad():
        source_est = sda(emg_grid_down.to(device))
    
    # Get baseline performance metrics before transform (mainly to evaluate initial decomp.)
    pred_dts, sils_base = utils.get_silohuette(source_est.detach().cpu().numpy())
    matches_base, rate_of_agreement_base, f1_scores_base, sensitivities_base, precisions_base,_ = utils.spike_matching(dts, pred_dts, fs=fsamp)
    print('RoA Training:', np.mean(rate_of_agreement_base))

# Test 30 randomly sampled spatial transformations
for trans_idx in range(Nt):
    Tx, Ty = float(Txs[trans_idx]), float(Tys[trans_idx])
    theta = float(thetas[trans_idx])
    print('Paremeters of Transform --> Tx:', Tx, 'Ty:', Ty, 'Theta:', theta)

    print('INITIALIZE SDA MODULE...')
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R).to(device) # create SAL-Decomposition model

    # Start the wandb run
    wandb.init(
        # set the wandb project where this run will be logged
        project="sda-gaussian-muaps",
        name=f'{opt}-{SNR}-{fxmax}-{trans_idx}',
        # mode='disabled',
    )

    # Keep track of experimental parameters of the run
    params = {'opt': opt, 'SNR':SNR, 'fxmax': fxmax,
                'Tx': Tx, 'Ty': Ty, 'theta': theta}
    params.update({
        'sils_base_avg': np.mean(sils_base), 'sils_base_std': np.std(sils_base),
        'f1_score_base_avg': np.mean(f1_scores_base), 'f1_score_base_std': np.std(f1_scores_base),
        'rate_of_agreement_base_avg': np.mean(rate_of_agreement_base), 'rate_of_agreement_base_std': np.std(rate_of_agreement_base),
        '#mu_matches_base': np.sum([r > 0.7 for r in rate_of_agreement_base])
        })

    with torch.no_grad():
        print('APPLY TRANSFORM...')
        emg_grid_test = utils.apply_affine(emg_grid.detach().cpu().clone(), Tx*sampfactor, Ty*sampfactor, theta, mode='bicubic')

        print('DOWNSAMPLING...')
        emg_grid_test = sutils.downsample_grid(emg_grid_test, sampfactor).to(device)

        print('POST-TRANSFORM SOURCE ESTIMATE')
        with torch.no_grad():
            sources_transform = sda(emg_grid_test)

        pred_dts, sils = utils.get_silohuette(sources_transform.detach().cpu().numpy())
        matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(dts, pred_dts, fs=fsamp)
        print('RoA Post-Transform:', np.mean(rate_of_agreement))

        params.update({'sils_transform_avg': np.mean(sils), 'sils_transform_std': np.std(sils),
                'f1_score_transform_avg': np.mean(f1_scores), 'f1_score_transform_std': np.std(f1_scores),
                'rate_of_agreement_transform_avg': np.mean(rate_of_agreement), 'rate_of_agreement_transform_std': np.std(rate_of_agreement),
                '#mu_matches_transform': np.sum([r > 0.7 for r in rate_of_agreement])
        })

        emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
        extended_emg = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
        inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
        sda.inv_cov = inv_cov_test

    # Optimization
    if opt == 'fit':
        sources, losses = utils.search_fit_sda(emg_grid_test.to(torch.float32), sda, base_loss, npoints=0, nepochs=nepochs, batch_size=batch_size, boundaries=bounds, lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)

    elif opt == 'search_fit':
        sources, losses = utils.search_fit_sda(emg_grid_test.to(torch.float32), sda, base_loss, npoints=2*nepochs, nepochs=nepochs//2, batch_size=batch_size, boundaries=bounds, lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)
    else: # search only
        sources, losses = utils.search_fit_sda(emg_grid_test.to(torch.float32), sda, base_loss, npoints=3*nepochs, nepochs=0, batch_size=batch_size, boundaries=bounds, lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)

    # Get learned transformations
    Tx_est, Ty_est = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2
    theta_est = np.pi*sda.sal.rot_theta[0].item()
    print('OPTIMAL PARAMETERS FOUND --> Tx:', Tx_est, 'Ty:', Ty_est, 'Theta:', theta_est)
    params.update({'Tx_est': Tx_est, 'Ty_est':Ty_est, 'theta_est': theta_est})

    # Estimating the efficacy of spatial adaptation
    theta1 = utils.get_theta(emg_grid_test.shape, Tx=Tx, Ty=Ty, theta=theta)
    theta2 = utils.get_theta(emg_grid_test.shape, Tx=Tx_est, Ty=Ty_est, theta=theta_est)
    distance = utils.get_distance(emg_grid_test.shape, theta1, theta2) # get distance in pixels between initial and final location
    wandb.log({'transformation_distance': distance})
    print("Average Post-Correction Distance (mm): ", 4*distance)

    # Create mask based on transformed coordinates being within convex
    print('RECOMPUTING MINIMALLY CROPPED INVERSE COVARIANCE MATRIX...')
    theta = sda.sal.get_affine_transform(input_shape=(H, W))
    theta = theta.repeat(1, 1, 1)
    transformed_grid = sda.sal.get_grid(theta, input_shape=(H, W))
    transformed_grid[:,:,:,0] = (W-1)*(1 + transformed_grid[:,:,:,0])/2
    transformed_grid[:,:,:,1] = (H-1)*(1 + transformed_grid[:,:,:,1])/2

    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid.detach().cpu(), xcrop_max=xcrop, ycrop_max=ycrop)
    sda.lcrop, sda.rcrop, sda.tcrop, sda.bcrop = lcrop, rcrop, tcrop, bcrop
    sda.crop_mask = sda.get_crop_mask()

    emg_grid_crop_test = emg_grid_test[:, :, tcrop:emg_grid_test.shape[2]-bcrop, lcrop:emg_grid_test.shape[3]-rcrop].clone()
    extended_emg = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)
    sda.inv_cov = inv_cov_test

    # Optimal predictions
    with torch.no_grad():
        sources = sda(emg_grid_test)

    # Performance metrics based on output sources
    pred_dts, sils = utils.get_silohuette(sources.detach().cpu().numpy())
    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(dts, pred_dts, fs=fsamp)
    print('RoA:', rate_of_agreement)

    params.update({'sils_avg': np.mean(sils), 'sils_std': np.std(sils),
                'f1_score_avg': np.mean(f1_scores), 'f1_score_std': np.std(f1_scores),
                'rate_of_agreement_avg': np.mean(rate_of_agreement), 'rate_of_agreement_std': np.std(rate_of_agreement),
                '#mu_matches_avg': np.sum([r > 0.7 for r in rate_of_agreement])
    })

    # Get new covariance matrix 
    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).to(torch.float32).T
    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg_test, reg=reg).to(torch.float32)

    # Refinement of MUs detected
    Nr = 10
    for idx in tqdm(range(Nr)):
        with torch.no_grad():
            sta_test = utils.get_sta_templates(extended_emg_test.clone(), pred_dts).to(torch.float32).to(device)
            sep_mat_test = sta_test @ inv_cov_test
            sources = (sep_mat_test @ extended_emg_test).T

        # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
        pred_dts, sils = utils.get_silohuette(sources)

        # Compute metrics
        matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(dts, pred_dts, old_matches=matches, fs=fsamp)
        print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.7 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")

    print(rate_of_agreement)
    print()

    params.update({'sils_refine_avg': np.mean(sils), 'sils_refine_std': np.std(sils),
                'f1_score_refine_avg': np.mean(f1_scores), 'f1_score_refine_std': np.std(f1_scores),
                'rate_of_agreement_refine_avg': np.mean(rate_of_agreement), 'rate_of_agreement_refine_std': np.std(rate_of_agreement),
                '#mu_matches_refine': np.sum([r > 0.7 for r in rate_of_agreement])
    })

    # Finish wandb run with all scores and parameters of the system
    wandb.log(params)
    wandb.finish()

