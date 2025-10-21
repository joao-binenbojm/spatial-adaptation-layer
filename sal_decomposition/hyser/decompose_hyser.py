import pandas as pd
import numpy as np
from scipy import signal
from sal_decomposition.utils import utils
import torch
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sal_decomposition.utils.grid_indexing import index_matrix4

def bandstop_filter(data, fsamp=2048): ## ACTUALLY BANDSTOP, NOT NOTCH
    '''Used to remove powerline interference and its multiples.'''
    sos = signal.butter(4, (45, 55), btype='bandstop', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def bandpass_filter(data, fsamp=2048):
    '''Used to maintain only information in relevant anatomical range of sEMG activity.'''
    sos = signal.butter(2, (20, 500), btype='bandpass', output='sos', fs=fsamp)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def notch_filter(data, fsamp=2048):
    '''Used to remove powerline interference at 50Hz.'''
    b, a = signal.iirnotch(50, Q=30, fs=fsamp)
    sos = signal.tf2sos(b, a)
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def fast_ica(X, n_components, R=16, max_iter=200, tol=1e-12):
    """
    Performs FastICA on the input data X.

    Args:
        X (torch.Tensor): Input data of shape (n_features, n_samples).
        n_components (int): The number of components to extract.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        torch.Tensor: The unmixing matrix W of shape (n_components, n_features).
    """
    T, C, H, W = X.shape
    device = X.device

    # 1. Center the data
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean

    # 2. Extend EMG data
    X_extended = utils.extend_emg_torch(X_centered.squeeze().reshape(X_centered.shape[0], -1), R=16).T

    # 3. Obtain whitening matrix
    print('WHITENING DATA...')
    X_whitened, whitening_matrix = utils.whitening_torch(X_extended, explained_var=1-1e-3)
    X_whitened, whitening_matrix = X_whitened.to(torch.float32), whitening_matrix.to(torch.float32)

    # 3. Initialize unmixing matrix W
    W = torch.randn(n_components, H*W*R, device=device)
    W = W / torch.norm(W, dim=1, keepdim=True)

    # 4. Iteratively update W
    print('OPTIMIZING SEPARATION VECTORS...')
    for i in tqdm(range(max_iter)):
        W_prev = W.clone()
        
        # g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
        g_u = torch.tanh(torch.matmul(W, X_whitened))
        g_prime_u = 1 - g_u**2

        W = torch.matmul(g_u, X_whitened.t()) / T - \
            torch.mean(g_prime_u, dim=1, keepdim=True) * W

          # SVD Orthogonalization (decorrelation) step
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        W = torch.matmul(U, Vh)
        
        # Check for convergence
        delta = torch.max(torch.abs(torch.abs(torch.sum(W * W_prev, dim=1)) - 1))
        if delta < tol:
            print(f"FastICA converged at iteration {i+1}")
            break
        else: # This else belongs to the for loop, executes if loop finishes without break
            print(f"FastICA reached max iterations ({max_iter}) without converging.")
            
    return W, X_whitened, whitening_matrix


## Load data and try and apply decomposition

if __name__ == '__main__':

    # Parameters
    subject = 3
    ied = 4
    mvc = 25
    session1 = 2
    session2 = 2

    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited'
    DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject}_edited'
    index_matrix = index_matrix4

    if subject == 1:
        DIR = os.path.join(DIR, f'{ied}mm')        


    with open('./sal_decomposition/healthy_surface/outlier_channels.json', 'r') as f:
        outliers = json.load(f) 
        
    file = f'S{subject}_{mvc}_Session{session1}_MUEdit_edited.mat'
    file2 = f'S{subject}_{mvc}_Session{session2}_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 10000
    bounds = [3, 3, 20*np.pi/180]
    # torch.set_default_dtype(torch.float64)
    print(os.listdir(DIR))

    # Load training data
    sgnl, edition = utils.open_mat_output(DIR, file)
    start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
    print('FILTERING TRAINING DATA...')
    emg = sgnl['data'][:, start:end]
    # emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg = notch_filter(bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
    # emg = bandpass_filter(emg, fsamp=fsamp)
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg_grid = utils.make_grid(emg, index_matrix)
    H, W = emg_grid.shape[2], emg_grid.shape[3]
    Nch = H*W
    # Compute outliers as channels average of neighbours
    print('HANDLING OUTLIER CHANNELS...')
    visible_outliers = np.zeros((H, W), dtype=np.bool_)
    outlier_coords = outliers[f"subject{subject}"][f"session{session1}"]
    indices = tuple(np.array(outlier_coords).T)
    visible_outliers[indices] = True
    # visible_outliers[11:14, -1] = True
    # visible_outliers[5, 1] = True
    # visible_outliers[2:13, 5:7] = True # Checking if this fixes things
    emg_grid = utils.handle_outliers(emg_grid, visible_outliers=visible_outliers)
    # emg_grid = emg_grid / (emg_grid.std() + 1e-12)
    # Load discharge times
    # dts = edition['Dischargetimes']
    # mu_dts = utils.squeeze_dts(dts)
    # mu_dts = utils.filter_dts(mu_dts, start, end)

    print('RUN DECOMPOSITION ALGORITHM FROM SCRATCH...')
    W, whitened_extended_emg, whitening_matrix = fast_ica(emg_grid, n_components=10, R=16)
    sources = (W @ whitened_extended_emg).T
    pred_dts, sils = utils.get_silohuette(sources)
    print()
    # matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts2, pred_dts, old_matches=official_matches, fs=fsamp)
        