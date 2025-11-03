import numpy as np
import scipy
from scipy.io import loadmat
from scipy import signal
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from math import floor
import wandb

from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2


if __name__ == '__main__':

    # Parameters
    ied = 4
    mvc = 25
    H, W = 26, 10
    index_matrix = index_matrix4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    nepochs = 250
    lr = 5e-3

    # Set pipeline parameters
    R = 16
    reg = 1e-1
    bounds = [2.5, 2.5, 15*np.pi/180]
    y_margin, x_margin = utils.out_of_bounds_pixels(26, 10, bounds[2])
    xcrop, ycrop = int(bounds[0] + floor(x_margin + 0.5)), int(bounds[1] + floor(y_margin + 0.5))

    with open('./sal_decomposition/healthy_surface/outlier_channels.json', 'r') as f:
        outliers = json.load(f) 

    for idx, subject in enumerate([0,1,2]):
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject+1}_edited'
        if subject == 0:
            DIR  = os.path.join(DIR, f'{ied}mm')

        for mvc in [25,50]:
            for ses1 in [0,1,2]:
                for ses2 in [0,1,2]:
                    if ses1 == ses2:
                        continue
                    
                    ###################### TEST LOADING SPECIFIC COMBINATION TO TEST WHETHER ANYTHING HAS CHANGED
                    DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject+1}_edited'
                    subject = 2
                    ses1, ses2 = 0, 1
                    #############################################################################################
                    # Initialize wandb run
                    run = wandb.init(
                        entity='jp2717-imperial-college-london',
                        project='sda-real-data-simulations',
                        name=f'sub={subject+1}_ses1={ses1+1}_ses2={ses2+1}mvc={mvc}',
                        config={'Subject': subject+1, 'Session1':ses1+1, 'Session2':ses2+1, 'MVC': mvc},
                        mode='disabled'
                    )

                    # Load both sessions from the same subject
                    file1 = f'S{subject+1}_{mvc}_Session{ses1+1}_MUEdit_edited.mat'
                    file2 = f'S{subject+1}_{mvc}_Session{ses2+1}_MUEdit_edited.mat'

                    # Load training data
                    sgnl, edition = utils.open_mat_output(DIR, file1)
                    start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
                    print('FILTERING TRAINING DATA...')
                    emg = sgnl['data']
                    emg = utils.bandstop_filter(utils.bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
                    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std(axis=1, keepdims=True) + 1e-12) # centering emg
                    emg_grid_original = utils.make_grid(emg, index_matrix4)
                    H, W = emg_grid_original.shape[2], emg_grid_original.shape[3]
                    Nch = H*W
                    # Compute outliers as channels average of neighbours
                    print('HANDLING OUTLIER CHANNELS...')
                    visible_outliers = np.zeros((H, W), dtype=np.bool_)
                    outlier_coords = outliers[f"subject{subject+1}"][f"session{ses1+1}"]
                    indices = tuple(np.array(outlier_coords).T)
                    visible_outliers[indices] = True
                    emg_grid_original = utils.handle_outliers(emg_grid_original, visible_outliers=visible_outliers)
                    emg_grid = emg_grid_original[start:end]

                    # Load discharge times
                    dts = edition['Dischargetimes']
                    mu_dts = utils.squeeze_dts(dts)
                    mu_dts = utils.filter_dts(mu_dts, start, end)
    
                    # Load test data
                    sgnl2, edition2 = utils.open_mat_output(DIR, file2)
                    start2, end2 = utils.get_target_boundaries(sgnl2['target'].squeeze())
                    emg2 = sgnl2['data']
                    print('FILTERING TEST DATA...')
                    emg2 = utils.bandstop_filter(utils.bandpass_filter(emg2, fsamp=fsamp), fsamp=fsamp)
                    emg2 = (emg2 - emg2.mean(axis=1, keepdims=True)) / (emg2.std() + 1e-12)
                    emg_grid_test_original = utils.make_grid(emg2, index_matrix)

                    print('HANDLING OUTLIER CHANNELS...')
                    visible_outliers = np.zeros((H, W), dtype=np.bool_)
                    outlier_coords = outliers[f"subject{subject+1}"][f"session{ses2+1}"]
                    indices = tuple(np.array(outlier_coords).T)
                    visible_outliers[indices] = True
                    emg_grid_test_original = utils.handle_outliers(emg_grid_test_original, visible_outliers=visible_outliers)
                    emg_grid_test = emg_grid_test_original[start2:end2]

                    # Load discharge times
                    dts2 = edition2['Dischargetimes']
                    mu_dts2 = utils.squeeze_dts(dts2)
                    mu_dts2 = utils.filter_dts(mu_dts2, start2, end2)
                    fr_test = utils.get_mean_firing_rate(mu_dts2, fs=fsamp)

                    # Get cropped inverse covariance
                    emg_grid_crop = emg_grid[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
                    extended_emg_crop = utils.extend_emg_torch(emg_grid_crop.squeeze().reshape(emg_grid_crop.shape[0], -1), R).T
                    inv_cov_train = utils.get_inv_cov_tikhonov(extended_emg_crop, reg=reg).to(torch.float32)

                    extended_emg = utils.extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
                    STA = utils.get_sta_templates(extended_emg, mu_dts).to(torch.float32)

                    # Initialize SDA module
                    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), STA=STA, inv_cov=inv_cov_train, xcrop=xcrop, ycrop=ycrop, extension_factor=R)

                    with torch.no_grad():
                        sources = sda(emg_grid)
                    pred_dts_train, sils = utils.get_silohuette(sources)
                    matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts, pred_dts_train, fs=fsamp)
                    print('RoA Training:', np.mean(rate_of_agreement))

                    params = {
                        'sils_base_avg': np.mean(sils), 'sils_base_std': np.std(sils),
                        'f1_score_base_avg': np.mean(f1_scores), 'f1_score_base_std': np.std(f1_scores),
                        'rate_of_agreement_base_avg': np.mean(rate_of_agreement), 'rate_of_agreement_base_std': np.std(rate_of_agreement),
                        '#mu_matches_base': np.sum([r > 0.7 for r in rate_of_agreement])
                    }

                    # Get base loss so we can understand how much sparsity relative to the training set/original decomposition
                    base_loss = utils.get_base_loss(emg_grid, sda.to(device), batch_size=batch_size, loss='negentropy', device=device)
            
                    # Get inverse covariance of the test grid, and determine the spatial transformation required for the STA templates to become optimal separation vectors
                    emg_grid_crop_test = emg_grid_test[:, :, ycrop:emg_grid_test.shape[2]-ycrop, xcrop:emg_grid_test.shape[3]-xcrop].clone()
                    extended_emg_crop_test = utils.extend_emg_torch(emg_grid_crop_test.squeeze().reshape(emg_grid_crop_test.shape[0], -1), R).T
                    inv_cov = utils.get_inv_cov_tikhonov(extended_emg_crop_test, reg=reg).to(torch.float32)
                    sda.inv_cov = inv_cov

                    loss_arr = utils.loss_sampling(emg_grid_test, sda.to(device), base_loss=base_loss, bounds=(2.5, 2.5), batch_size=batch_size, num_points=20, loss='negentropy', device=device)
                    sources, losses = utils.search_fit_sda(emg_grid_test.clone(), sda=sda.to(device), base_loss=base_loss, batch_size=batch_size, npoints=2*nepochs, nepochs=nepochs, lr=5e-3, boundaries=bounds, loss='negentropy', device=device)

                    # Get new inverse covariance
                    sda = sda.to('cpu')
                    sda.sal.mode = 'bicubic'

                    # Get minimum distance between original and transformed grid
                    Tx_est, Ty_est, theta_est = (W-1)*sda.sal.xshift[0].item()/2, (H-1)*sda.sal.yshift[0].item()/2, sda.sal.rot_theta[0].item()*np.pi
                    print(f"Tx_est: {Tx_est}, Ty_est: {Ty_est}, theta_est: {theta_est}")

                    # Create mask based on transformed coordinates being within convex 
                    # theta1 = utils.get_theta(emg_grid_test.shape, Tx=Tx_est_est, Ty=Ty, theta=theta_est)
                    # theta2 = utils.get_theta(emg_grid_test.shape, Tx=Tx_est, Ty=Ty_est, theta=theta_est)
                    # distance = utils.get_distance(emg_grid_test.shape, theta1, theta2) # get distance in pixels between initial and final location
                    # wandb.log({'transformation_distance': distance})
                    # print("Average Post-Correction Distance (mm): ", 4*distance)

                    # Create mask based on transformed coordinates being within convex
                    print('RECOMPUTING MINIMALLY CROPPED INVERSE COVARIANCE MATRIX...')
                    theta = sda.sal.get_affine_transform(input_shape=(H, W))
                    theta = theta.repeat(1, 1, 1)
                    transformed_grid = sda.sal.get_grid(theta, input_shape=(H, W))
                    transformed_grid[:,:,:,0] = (W-1)*(1 + transformed_grid[:,:,:,0])/2
                    transformed_grid[:,:,:,1] = (H-1)*(1 + transformed_grid[:,:,:,1])/2

                    lcrop, rcrop, bcrop, tcrop = utils.get_min_conservative_crop((H, W), transformed_grid, xcrop_new=xcrop, ycrop_new=ycrop)
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

                    # Get new covariance matrix based on statistics of full test data
                    extended_emg_test = utils.extend_emg_torch(emg_grid_test.squeeze().reshape(emg_grid_test.shape[0], -1), R).T
                    inv_cov_test = utils.get_inv_cov_tikhonov(extended_emg, reg=reg).to(torch.float32)

                    # Refinement of MUs detected
                    Nr = 10
                    official_matches = dict(matches)
                    for idx in tqdm(range(Nr)):
                        with torch.no_grad():
                            sta_test = utils.get_sta_templates(extended_emg_test.clone(), pred_dts)
                            sep_mat_test = sta_test @ inv_cov_test
                            sources = (sep_mat_test @ extended_emg_test).T

                        # Recompute predicted discharges and compute performances: only consider original matches made, not new ones!!
                        pred_dts, sils = utils.get_silohuette(sources)
                        matches, rate_of_agreement, f1_scores, sensitivities, precisions, zscores = utils.spike_matching(mu_dts2, pred_dts, old_matches=official_matches, fs=fsamp)
                        print(f"Refinement Step #{idx+1} --> #MU matches: {np.sum([r > 0.7 for r in rate_of_agreement])}, RoA: {np.mean(rate_of_agreement)}, F1-Score: {np.mean(f1_scores)}")
                        
                    print(rate_of_agreement)
                    params = {
                        'sils_refine_avg': np.mean(sils), 'sils_refine_std': np.std(sils),
                        'f1_score_refine_avg': np.mean(f1_scores), 'f1_score_refine_std': np.std(f1_scores),
                        'rate_of_agreement_refine_avg': np.mean(rate_of_agreement), 'rate_of_agreement_refine_std': np.std(rate_of_agreement),
                        '#mu_matches_refine': np.sum([r > 0.7 for r in rate_of_agreement])
                    }
