import numpy as np
from scipy import signal
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from math import floor

# from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2
import wandb


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    lr = 5e-3
    loss='negentropy'
    Nt = 50
    nepochs = 250
    bounds = np.array([2.5, 2.5, 15*np.pi/180])

    y_margin, x_margin = utils.out_of_bounds_pixels(26, 10, bounds[2])
    xcrop, ycrop = int(bounds[0] + floor(x_margin + 0.5)), int(bounds[1] + floor(y_margin + 0.5))
    
    R = 16
    reg = 1e-1
    
    for sub_idx in [0,1,2]: 
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{sub_idx+1}_edited'
        if sub_idx == 0:
            DIR = os.path.join(DIR, '4mm')

        for mvc in [25, 50]:
            for ses_idx in [0,1,2]:
                file = f"S{sub_idx+1}_{mvc}_Session{ses_idx+1}_MUEdit_edited.mat" 
                sgnl, edition = utils.open_mat_output(DIR, file)
                start, end = utils.get_target_boundaries(sgnl['target'].squeeze())

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
                # Crop observations and get new sep_mat
                print(f'CROPS: XCROP: {xcrop}, YCROP: {ycrop}')
                emg_grid_crop_train = emg_grid[:, :, ycrop:emg_grid.shape[2]-ycrop, xcrop:emg_grid.shape[3]-xcrop].clone()
                extended_emg_crop_train = utils.extend_emg_torch(emg_grid_crop_train.squeeze().reshape(emg_grid_crop_train.shape[0], -1), R).T
                inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop_train, reg=reg).to(torch.float32)

                # Test that masking channels is a valid solution
                print('TESTING MASKING CHANNELS...')
                extended_emg_train = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
                STA = utils.get_sta_templates(extended_emg_train, mu_dts).to(torch.float32) # gets STA templates from Session 1 from the raw data

                # Loop over extension factor, explained variance, and transformations
                for trans_idx in range(Nt): # Nt random transformations
                    Tx, Ty, theta = torch.tensor(np.random.uniform(-bounds, bounds)).to(torch.float32)
                    print(f"Transformation parameters: Tx: {Tx}, Ty: {Ty}, rot_theta: {theta}")

                    # Initialize wandb run
                    run = wandb.init(
                        entity='jp2717-imperial-college-london',
                        project='sda-real-data-simulations',
                        name=f'sub_{sub_idx+1}_ses_{ses_idx+1}_mvc_{mvc}',
                        config={'Subject': sub_idx+1, 'Session':ses_idx+1, 'MVC': mvc},
                        mode='disabled'
                    )

                    params = {"Tx": Tx, "Ty": Ty, "theta": theta}

                    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)
                    base_loss = utils.get_base_loss(emg_grid, sda, batch_size=batch_size, loss=loss, device='cpu')

                    # Test on original grid with non-regularized whitening matrix
                    with torch.no_grad():
                        sources = sda(emg_grid)
                    pred_dts, sils = utils.get_silohuette(sources)
                    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                    print('ROA Training:', np.mean(rate_of_agreement))
                    print('#mu_train:', sum([roa > 0.7 for roa in rate_of_agreement]))

                    params.update({
                        'sils_base_avg': np.mean(sils), 'sils_base_std': np.std(sils),
                        'f1_score_base_avg': np.mean(f1_scores), 'f1_score_base_std': np.std(f1_scores),
                        'rate_of_agreement_base_avg': np.mean(rate_of_agreement), 'rate_of_agreement_base_std': np.std(rate_of_agreement),
                        '#mu_matches_base': np.sum([r > 0.7 for r in rate_of_agreement])
                        })

                    # Add optimal parameters for testing
                    with torch.no_grad():
                        sda.sal.xshift[0].copy_(2*Tx/(W-1))
                        sda.sal.yshift[0].copy_(2*Ty/(H-1))
                        sda.sal.rot_theta[0].copy_(theta/np.pi)
                        emg_grid_test = sda.sal(emg_grid) # apply spatial transformation to get simulated data
                    
                    # Compute performance post-transform, before adaptation
                    with torch.no_grad():
                        sources = sda(emg_grid_test)
                    pred_dts, sils = utils.get_silohuette(sources)
                    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                    print('ROA Post-transform:', np.mean(rate_of_agreement))
                    print('#mu_transform:', sum([roa > 0.7 for roa in rate_of_agreement]))

                    params.update({
                        'sils_transform_avg': np.mean(sils), 'sils_transform_std': np.std(sils),
                        'f1_score_transform_avg': np.mean(f1_scores), 'f1_score_transform_std': np.std(f1_scores),
                        'rate_of_agreement_transform_avg': np.mean(rate_of_agreement), 'rate_of_agreement_transform_std': np.std(rate_of_agreement),
                        '#mu_matches_transform': np.sum([r > 0.7 for r in rate_of_agreement])
                    })

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

                    # Fit SDA model to determine optimal spatial transformation
                    sources, losses = utils.search_fit_sda(emg_grid_test, sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=2*nepochs, nepochs=nepochs, lr=lr, boundaries=bounds, device='cuda', loss=loss)
                    sda.to('cpu') # send back to CPU

                    # Log parameters
                    Tx_est = (W-1)*sda.sal.xshift[0].item()/2
                    Ty_est = (H-1)*sda.sal.yshift[0].item()/2
                    theta_est = np.pi*sda.sal.rot_theta[0].item()
                    wandb.log({'Tx_est': Tx_est, 'Ty_est': Ty_est, 'theta_est': theta_est})

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

                    pred_dts, sils = utils.get_silohuette(sources.to('cpu'))
                    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, fs=fsamp)
                    print("ROA Test:", np.mean(rate_of_agreement))
                    print('#mu_test: ', sum([roa > 0.7 for roa in rate_of_agreement]))
                    params.update({
                        'sils_avg': np.mean(sils), 'sils_std': np.std(sils),
                        'f1_score_avg': np.mean(f1_scores), 'f1_score_std': np.std(f1_scores),
                        'rate_of_agreement_avg': np.mean(rate_of_agreement), 'rate_of_agreement_std': np.std(rate_of_agreement),
                        '#mu_matches': np.sum([r > 0.7 for r in rate_of_agreement])
                    })

                    # Get new covariance matrix 
                    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg_test, reg=1e-3) #.to(torch.float32)

                    # Refinement of MUs detected
                    Nr = 10
                    for idx in tqdm(range(Nr)):
                        with torch.no_grad():
                            sta_test = utils.get_sta_templates(extended_emg_test.clone(), pred_dts)
                            sep_mat_test = sta_test @ inv_cov_test
                            sources = (sep_mat_test @ extended_emg_test).T

                        # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
                        pred_dts, sils = utils.get_silohuette(sources)

                        # Compute metrics
                        matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts, old_matches=matches, fs=fsamp)
                        print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.7 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}, Precision: {np.mean(precisions)}, Sensitivity: {np.mean(sensitivities)}")

                    print(rate_of_agreement)
                    print()                
                    
                    params.update({
                        'sils_refine_avg': np.mean(sils), 'sils_refine_std': np.std(sils),
                        'f1_score_refine_avg': np.mean(f1_scores), 'f1_score_refine_std': np.std(f1_scores),
                        'rate_of_agreement_refine_avg': np.mean(rate_of_agreement), 'rate_of_agreement_refine_std': np.std(rate_of_agreement),
                        '#mu_matches_refine': np.sum([r > 0.7 for r in rate_of_agreement])
                    })

                    wandb.log(params)
                    wandb.finish()