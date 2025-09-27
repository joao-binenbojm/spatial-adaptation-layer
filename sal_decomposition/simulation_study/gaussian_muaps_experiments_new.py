import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import wandb
import pandas as pd

import sal_decomposition.utils.simulation_utils as sutils
from sal_decomposition.utils import utils
from sal_decomposition.sda import SpatialDecompositionAdaptationOld

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
SNRs = [30, 15, 5, 1]
fxmaxs = [187.5, 156.25, 125, 93.75, 62.5] # m^-1
# fxmaxs = [93.75]
opts = ['search_fit', 'search', 'fit'] # whether to only train, only search, or search and fit

# Fixed simulation parameters
fsamp = 2000 # Hz
fsx = 250 # m^-1
duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
H, W, L = 26, 10, 50
# H, W, L = 5, 5, 20
R = 16
sampfactor=14
delay = (torch.floor(torch.tensor([L + R])/2) - 1).to(torch.int) # delay introduced by causality of triggering process

# Training params
batch_size = 10000
nepochs=120
lr = 5e-3
loss = 'kurtosis' # loss function for optimization
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

# Create range of spatial transformations to be used equally for every single condition
Nt = 50
Txs, Tys = np.random.uniform(-1.2, 1.2, size=Nt), np.random.uniform(-1.2, 1.2, size=Nt)
thetas = np.random.uniform(-20*np.pi/180, 20*np.pi/180, size=Nt)
xscales, yscales = np.random.uniform(0.8, 1.2, size=Nt), np.random.uniform(0.8, 1.2, size=Nt)
# xscales, yscales = np.random.uniform(1.0, 1.0, size=Nt), np.random.uniform(1.0, 1.0, size=Nt)


for fxmax in tqdm(fxmaxs):
    for SNR in SNRs:
        for opt in opts:
            # If in checklist, already run, continue to next condition
            if (fxmax, SNR, opt) in checklist:
                continue
            with torch.no_grad():
                exp = SDAExperiment()
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
                # emg = exp.generate_emg(spts, muaps, R=R) # make synthetic EMG from simulated MUAPs and spike trains
                # Get separation vector
                # muaps_down = exp.downsample_muaps(muaps, sampfactor) # downsample MUAPs
                muaps_down = sutils.downsample_muaps(muaps, sampfactor).to('cpu') # downsample MUAPs
                # B = exp.get_separation_vectors(muaps_down, R=R)
                B = sutils.get_separation_vectors(muaps_down, R=R, delay=delay)
                
                # noisy_emg = exp.add_noise(emg, SNR) # add noise to synthetic signal
                noisy_emg = sutils.add_noise(emg, SNR) # add noise to synthetic signal
                noisy_emg = (noisy_emg - noisy_emg.mean()) / (noisy_emg.std() + 1e-12) # standardize noisy emg
                # emg_grid = exp.make_grid(noisy_emg) # reshape into EMG grid
                emg_grid = sutils.simulation_make_grid(noisy_emg) # reshape into EMG grid
                # emg_grid_down = exp.downsample_grid(emg_grid, sampfactor) # downsample EMG grid
                emg_grid_down = sutils.downsample_grid(emg_grid, sampfactor) # downsample EMG grid

                # Right multiply inverse covariance matrix
                extended_emg = utils.extend_emg_torch(emg_grid_down.squeeze().reshape(emg_grid_down.shape[0], -1), R).T
                inv_cov = utils.get_inv_cov_torch(extended_emg, explained_var=1-1e-2).to(torch.float32)
                sep_mat = B @ inv_cov

                print('GET SEPARATION VECTORS & WHITENING...')
                sda = SpatialDecompositionAdaptationOld(grid_shape=(H, W), extension_factor=R, sep_mat=sep_mat).to(device) # create SAL-Decomposition model
                base_loss = utils.get_base_loss(emg_grid_down, sda, batch_size=batch_size, loss='kurtosis', device=device) # get baseline loss
                print('BASELINE LOSS:', base_loss)

                # with torch.no_grad():
                #     source_est = sda(emg_grid_down)
                with torch.no_grad():
                    source_est = sda(emg_grid_down.to(device))

                # source_est = exp.process_sep_mat(emg_grid_down, B, R=R)
                # exp.get_base_loss(emg_grid_down.to(torch.float32), loss=loss, device=device) # get baseline loss
                
                # Get baseline performance metrics before transform (mainly to evaluate initial decomp.)
                # pred_dts, sils_base = exp.get_silohuette(source_est.detach().cpu().numpy().T)
                pred_dts, sils_base = utils.get_silohuette(source_est.detach().cpu().numpy())
                # scores_base = exp.spike_scores(dts, pred_dts)
                matches, f1_scores, sensitivities, precisions = utils.spike_matching(dts, pred_dts, fs=fsamp)
                print('F1 Scores Training:', np.mean(f1_scores))
                # wandb.log({'f1_train': np.mean(f1_scores)})
                # wandb.log({'#mu_train': sum([f1_score > 0.8 for f1_score in f1_scores])})

            # Test 30 randomly sampled spatial transformations
            for trans_idx in range(Nt):
                Tx, Ty = float(Txs[trans_idx]), float(Tys[trans_idx])
                theta = float(thetas[trans_idx])
                xscale, yscale = float(xscales[trans_idx]), float(yscales[trans_idx])

                # Start the wandb run
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="sal-decomposition-simulations-new",
                    name=f'{opt}-{SNR}-{fxmax}',
                    # mode='disabled',
                )
            
                # Keep track of experimental parameters of the run
                params = {'opt': opt, 'SNR':SNR, 'fxmax': fxmax,
                            'Tx': Tx, 'Ty': Ty, 'theta': theta, 'xscale': xscale, 'yscale': yscale}
                params.update({
                    'sils_base_avg': np.mean(sils_base), 'sils_base_std': np.std(sils_base),
                    'sensitivity_base_avg': np.mean(sensitivities), 'sensitivity_base_std': np.std(sensitivities),
                    'precision_base_avg': np.mean(precisions), 'precision_base_std': np.std(precisions),
                    'f1_score_base_avg': np.mean(f1_scores), 'f1_score_base_std': np.std(f1_scores)

                    })

                with torch.no_grad():
                    print('APPLY TRANSFORM...')
                    # emg_grid_transform = exp.apply_affine(emg_grid.detach().clone(), Tx, Ty, theta, xscale, yscale, sampfactor)
                    emg_grid_transform = utils.apply_affine(emg_grid.detach().cpu().clone(), Tx*sampfactor, Ty*sampfactor, theta, xscale, yscale, mode='bicubic')

                    print('DOWNSAMPLING...')
                    # emg_grid_transform = exp.downsample_grid(emg_grid_transform, sampfactor)
                    emg_grid_transform = sutils.downsample_grid(emg_grid_transform, sampfactor).to(device)

                    # print('CENTERING...')
                    # emg_grid_down = emg_grid_down.to(device)
                    # mean = (emg_grid_down.mean(dim=0, keepdim=True) + emg_grid_transform.mean(dim=0, keepdim=True))/2
                    # emg_grid_down, emg_grid_transform = emg_grid_down - mean, emg_grid_transform - mean 

                    print('POST-TRANSFORM SOURCE ESTIMATE')
                    # sources_transform = exp.get_source_estimate(emg_grid_transform)
                    with torch.no_grad():
                        sources_transform = sda(emg_grid_transform.to(torch.float32))

                    # pred_dts, sils_transform = exp.get_silohuette(sources_transform.detach().cpu().numpy().T)
                    pred_dts, sils_transform = utils.get_silohuette(sources_transform.detach().cpu().numpy())
                    # scores_transform = exp.spike_scores(dts, pred_dts)
                    matches, f1_scores_transform, sensitivities_transform, precisions_transform = utils.spike_matching(dts, pred_dts, fs=fsamp)
                    print('F1 Scores Post-Transform:', np.mean(f1_scores_transform))
                    params.update({
                    'sensitivity_transform_avg': np.mean(sensitivities), 'sensitivity_transform_std': np.std(sensitivities),
                    'precision_transform_avg': np.mean(precisions), 'precision_transform_std': np.std(precisions),
                    'f1_score_transform_avg': np.mean(f1_scores), 'f1_score_transform_std': np.std(f1_scores)
                    })

                # Optimization
                if opt == 'fit':
                    sources, losses = utils.search_fit_sda(emg_grid_transform.to(torch.float32), sda, base_loss, npoints=0, nepochs=nepochs, batch_size=2048, boundaries=(1.2, 1.2, 20*np.pi/180, 0.01, 0.01), lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)

                elif opt == 'search_fit':
                    sources, losses = utils.search_fit_sda(emg_grid_transform.to(torch.float32), sda, base_loss, npoints=2*nepochs, nepochs=nepochs//2, batch_size=2048, boundaries=(1.2, 1.2, 20*np.pi/180, 0.01, 0.01), lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)
                else: # search only
                    sources, losses = utils.search_fit_sda(emg_grid_transform.to(torch.float32), sda, base_loss, npoints=3*nepochs, nepochs=0, batch_size=2048, boundaries=(1.2, 1.2, 20*np.pi/180, 0.01, 0.01), lr=lr, device=device, loss=loss, plot=0, frozen_sep_mat=True)

                # Get learned transformations
                Tx_opt, Ty_opt = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2
                theta_opt = np.pi*sda.sal.rot_theta[0].item()
                xscale_opt, yscale_opt = sda.sal.xscale[0].item(), sda.sal.yscale[0].item()

                params.update({'Tx_opt': Tx_opt, 'Ty_opt':Ty_opt, 'theta_opt': theta_opt,
                            'xscale_opt':xscale_opt, 'yscale_opt': yscale_opt})

                # Performance metrics based on output sources
                # pred_dts, sils = exp.get_silohuette(sources.detach().cpu().numpy())
                pred_dts, sils = utils.get_silohuette(sources.detach().cpu().numpy())
                # scores = exp.spike_scores(dts, pred_dts)
                matches, f1_scores, sensitivities, precisions = utils.spike_matching(dts, pred_dts, fs=fsamp)
                print('SILS:', sils)
                print()
                print('F1-SCORES:', f1_scores)

                params.update({'sils_avg': np.mean(sils), 'sils_std': np.std(sils),
                            'sensitivity_avg': np.mean(sensitivities), 'sensitivity_std': np.std(sensitivities),
                            'precision_avg': np.mean(precisions), 'precision_std': np.std(precisions),
                            'f1_score_avg': np.mean(f1_scores), 'f1_score_std': np.std(f1_scores)
                })

                # Finish wandb run with all scores and parameters of the system
                wandb.log(params)
                wandb.finish()

            del params, muaps, dts, spts, emg, emg_grid, emg_grid_transform # delete params before next step of sims
            
