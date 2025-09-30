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
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2
import wandb

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10240
    bounds = np.array([2.0, 2.0, 20*np.pi/180, 1.1, 1.1])

    for sub_idx in [1]:#range(2):
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'
        if sub_idx == 0: DIR = os.path.join(DIR, '4mm')

        for mvc in [25, 50]:
            for ses_idx in [2]:#range(3):
                # file = f'S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat'
                file = f"S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat" 

                signal, edition = utils.open_mat_output(DIR, file)
                start, end = utils.get_target_boundaries(signal['target'].squeeze())

                # Apply filters to data and reshape into desired shape
                print('FILTER DATA...')
                emg = signal['data'][:, start:end]
                emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
                emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)
                emg_grid = utils.make_grid(emg, index_matrix4).to(torch.float32)
                # emg_grid = emg_grid[:,:,:emg_grid.shape[2]//2, :emg_grid.shape[3]//2]
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

                # Loop over extension factor, explained variance, and transformations
                for R in [16]:#[8, 16, 32]:
                    for explained_var in [1-1e-4]:#[1-1e-2, 1-1e-3, 1-1e-4, 1-1e-6]:
                        for trans_idx in range(30): # thirty random transformations
                            min_bounds = [-b if b_idx < 3 else 1/b for b_idx, b in enumerate(bounds)]
                            # Tx, Ty, theta, xscale, yscale = torch.tensor(np.random.uniform(min_bounds, bounds)).to(torch.float32) # sample random transformation parameters
                            Tx, Ty, theta = torch.tensor([2.0, -2.0, 0.0]) #, 1.0, 1.0])
                            # Tx, Ty, theta = torch.tensor(np.random.uniform(-bounds[:-2], bounds[:-2])).to(torch.float32)
                            print(f"Transformation parameters: Tx: {Tx}, Ty: {Ty}, rot_theta: {theta}")#, xscale: {xscale}, yscale: {yscale}")

                            # Initialize wandb run
                            run = wandb.init(
                                entity='jp2717-imperial-college-london',
                                project='real-data-simulations',
                                name=f'sub_{sub_idx+1}_ses_{ses_idx+1}_mvc_{mvc}',
                                config={'Subject': sub_idx+1, 'Session':ses_idx+1, 'MVC': mvc,
                                        'Extension Factor': R, 'Explained Variance': explained_var, 
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
                            inv_cov = utils.get_inv_cov_torch(extended_emg, explained_var=explained_var).to(torch.float32)
                            STA = utils.get_sta_templates(extended_emg, mu_dts).to(torch.float32) # gets STA templates from Session 1 from the raw data

                            sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov, extension_factor=R)
                            base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss='kurtosis', device='cpu')

                            # Test on original grid with non-regularized whitening matrix
                            with torch.no_grad():
                                source_est = sda(emg_grid)
                            pred_dts, sils = utils.get_silohuette(source_est)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print('F1 Scores Training:', np.mean(f1_scores))
                            print('#mu_train:', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_train': np.mean(f1_scores)})
                            wandb.log({'#mu_train': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Add optimal parameters for testing
                            with torch.no_grad():
                                sda.sal.xshift[0].copy_(2*Tx/(W-1))
                                sda.sal.yshift[0].copy_(2*Ty/(H-1))
                                sda.sal.rot_theta[0].copy_(theta/np.pi)
                                # sda.sal.xscale[0].copy_(xscale)
                                # sda.sal.yscale[0].copy_(yscale)
                                # sda.sal.xshift[0].copy_(2*-1.5/(W-1))
                                # sda.sal.yshift[0].copy_(2*1.8/(H-1))
                                # Get simulated spatial perturbation
                                emg_grid_test = sda.apply_affine(emg_grid)
                                # emg_grid = sda.apply_affine(emg_grid_test, inverse=True) # invert spatial transformation to get zeroed out and mixedl channels

                            # Test on optimal inverse transformation
                            extended_emg = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                            inv_cov = utils.get_inv_cov_torch(extended_emg, explained_var=explained_var).to(torch.float32)
                            sda.inv_cov = inv_cov # update inverse covariance
                            
                            # LOSS SAMPLING TO UNDERSTAND WHAT'S GOING ON
                            # utils.loss_sampling(emg_grid_test, sda, base_loss=base_loss, T=[-1.5, 1.8], bounds=[2.0, 2.0], batch_size=batch_size)

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
                                sda.sal.xscale[0].copy_(1.0)
                                sda.sal.yscale[0].copy_(1.0)

                            # Fit SDA model to determine optimal spatial transformation
                            sources, losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=2000, nepochs=0, lr=1e-3, boundaries=bounds[:-2], device='cuda')

                            # Log parameters
                            wandb.log({'Tx_est':(W-1)*sda.sal.xshift[0].item()/2, 'Ty_est':(H-1)*sda.sal.yshift[0].item()/2, 'theta_est':np.pi*sda.sal.rot_theta[0].item()})

                            pred_dts, sils = utils.get_silohuette(sources.to('cpu'))
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            print('#mu_test: ', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_test': np.mean(f1_scores)})
                            wandb.log({'#mu_test': sum([f1_score > 0.8 for f1_score in f1_scores])})

                            # Get sep mat based on real test data
                            print('Refining STA templates based on predicted discharge times...')
                            extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                            inv_cov_test = utils.get_inv_cov_torch(extended_emg_test, explained_var=explained_var)
                            STA = utils.get_sta_templates(extended_emg_test, pred_dts)
                            sep_mat_test = STA @ inv_cov_test
                            with torch.no_grad():
                                sources = (sep_mat_test @ extended_emg_test).T

                            pred_dts, sils = utils.get_silohuette(sources)
                            matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                            print(f1_scores)
                            print('#mu_refine:', sum([f1_score > 0.8 for f1_score in f1_scores]))
                            wandb.log({'f1_refine': np.mean(f1_scores)})
                            wandb.log({'#mu_refine': sum([f1_score > 0.8 for f1_score in f1_scores])})
                            wandb.finish()