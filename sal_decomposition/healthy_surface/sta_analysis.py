import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg, get_silohuette, maxk, bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from loss_functions import KurtosisLoss, NegentropyLoss
from sklearn.cluster import KMeans

# Arnault's matrix reshaping
index_matrix = np.array([[63, 38, 37, 12, 11, 63, 38, 37, 12, 11], # ankle
                [62, 39, 36, 13, 10, 62, 39, 36, 13, 10],
                [61, 40, 35, 14,  9, 61, 40, 35, 14,  9],
                [60, 41, 34, 15,  8, 60, 41, 34, 15,  8],
                [59, 42, 33, 16,  7, 59, 42, 33, 16,  7],
                [58, 43, 32, 17,  6, 58, 43, 32, 17,  6],
                [57, 44, 31, 18,  5, 57, 44, 31, 18,  5],
                [56, 45, 30, 19,  4, 56, 45, 30, 19,  4],
                [55, 46, 29, 20,  3, 55, 46, 29, 20,  3],
                [54, 47, 28, 21,  2, 54, 47, 28, 21,  2],
                [53, 48, 27, 22,  1, 53, 48, 27, 22,  1],
                [52, 49, 26, 23,  0, 52, 49, 26, 23,  0],
                [51, 50, 25, 24,  0, 51, 50, 25, 24,  0],
                [0, 24, 25, 50, 51,  0, 24, 25, 50, 51],
                [0, 23, 26, 49, 52,  0, 23, 26, 49, 52],
                [1, 22, 27, 48, 53,  1, 22, 27, 48, 53],
                [2, 21, 28, 47, 54,  2, 21, 28, 47, 54],
                [3, 20, 29, 46, 55,  3, 20, 29, 46, 55],
                [4, 19, 30, 45, 56,  4, 19, 30, 45, 56],
                [5, 18, 31, 44, 57,  5, 18, 31, 44, 57],
                [6, 17, 32, 43, 58,  6, 17, 32, 43, 58],
                [7, 16, 33, 42, 59,  7, 16, 33, 42, 59],
                [8, 15, 34, 41, 60,  8, 15, 34, 41, 60],
                [9, 14, 35, 40, 61,  9, 14, 35, 40, 61],
                [10, 13, 36, 39, 62, 10, 13, 36, 39, 62],
                [11, 12, 37, 38, 63, 11, 12, 37, 38, 63]]) # knee

# In the order of the cables, it is
# GRID 4    GRID 3
# GRID 1    GRID 2
# So taking the 256 signals in signal.data as input, one must reshape in
# the following way:

index_matrix[13:26,5:10] =  index_matrix[13:26,5:10] + 64 
index_matrix[0:13,5:10] = index_matrix[0:13,5:10] + 64 + 64 
index_matrix[0:13,0:5] = index_matrix[0:13,0:5] + 64 + 64 + 64   


def get_inv_cov_torch(signal, explained_var=0.99):
    ''' Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization all in PyTorch (compute in torch.float64 to avoid precision issues). '''
    cov_mat = torch.cov(signal.to(torch.float64))
    evalues, evectors  = torch.linalg.eigh(cov_mat)
    sorted_idxs = torch.argsort(evalues, descending=True)
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    cum_explained_var = evalues.cumsum(dim=0) / evalues.sum()
    evalues, evectors = evalues[cum_explained_var <= explained_var], evectors[:, cum_explained_var <= explained_var]
    inv_cov = evectors @ torch.diag(1 / (evalues)) @ evectors.T
    return inv_cov

def open_mat_output(DIR, name):
    ''' Open data from Emanuele's files so that it can be further processed'''
    arr = loadmat(os.path.join(DIR, name), struct_as_record=False, mat_dtype=True)
    
    # Load acquisition signal
    signal, edition = {}, {}
    for attr in dir(arr['signal'][0,0]):
        if attr[0] != '_': # only data relevant attributes
            signal[attr] = getattr(arr['signal'][0,0], attr)

    for attr in dir(arr['edition'][0,0]):
        if attr[0] != '_': # only data relevant attributes
            edition[attr] = getattr(arr['edition'][0,0], attr)
    
    return signal, edition

def get_target_boundaries(target, threshold=0.8):
    '''Takes in signal and edition dictionaries and returns the data and edition dictionaries with the target timestamps only.'''
    target_max = np.max(target)
    # Find rising and falling edges of target signal
    rising_edge = np.where(target >= threshold*target_max)[0][0]
    falling_edge = np.where(target >= threshold*target_max)[0][-1]
 
    return rising_edge, falling_edge

def filter_dts(dts, start, end):
    '''Takes in discharge times, removes dts outside of bounds, and accounts for target delay.'''
    for idx in range(len(dts)):
        dts[idx] = dts[idx][(dts[idx] < end) & (dts[idx] > start)] # filter for within target boundaries
        dts[idx] = dts[idx] - start # account for target delay
    return dts

def squeeze_dts(dts):
    '''Takes in discharge times across motor units in an array of shape (4, Nmax), 
    being the maximum number of MUs per subgrid (4) Return a list where each elemnt is an array of discharge times. '''
    # Convert discharge times array into list of arrays, removing zeros/padding
    mu_dts = []
    for idx in range(dts.shape[0]):
        for jdx in range(dts.shape[1]):
            # Get non-zero discharge times for this MU
            valid_dts = dts[idx, jdx]
            if valid_dts.shape[1] > 0:  # Only add if there are valid discharge times
                mu_dts.append(valid_dts.squeeze())
    return mu_dts

def extend_emg_torch(emg, R):
    '''Extend the original EMG batch given extension factor. Centered around the curren time.'''
    device = emg.device
    nchans = emg.shape[1]
    extended_emg = torch.zeros((emg.shape[0] + R - 1, nchans*R)).to(torch.float64).to(device)
    for idx in range(R):
        extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
    return extended_emg

def get_sep_mat_torch(extended_emg, dts):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    N = len(dts) # number of MUs
    sep_mat = torch.zeros((N, extended_emg.shape[1])).to(torch.float64)
    for idx in range(N):
        sep_mat[idx, :] = (extended_emg[dts[idx].astype(int), :]).mean(dim=0)
    return sep_mat


if __name__ == '__main__':
    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/2mm'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    L = 100
    R = 8
    file = 'S1_25_2mm_Session1_MUEdit_edited.mat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048
    batch_size = 16384
    print(os.listdir(DIR))
    signal, edition = open_mat_output(DIR, file)
    start, end = get_target_boundaries(signal['target'].squeeze())
    torch.set_default_dtype(torch.float64)

    # Apply filters to data and reshape into desired shape
    print('FILTER DATA...')
    emg = signal['data'][:, start:end]
    Nch = emg.shape[0]  
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-9) # centering emg
    emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)


    # Get inverse covariance matrix
    # extended_emg_template = np.zeros((R*Nch, emg.shape[1] + R - 1))
    # extended_emg = torch.tensor(extend_emg(extended_emg_template, emg, R))#.to(torch.float32)
    # inv_cov = get_inv_cov(extended_emg, explained_var=1.0-1e-14)


    # # Get separation matrix
    extended_emg = extend_emg_torch(torch.tensor(emg.T), L)
    dts = edition['Dischargetimes']
    mu_dts = squeeze_dts(dts)
    mu_dts = filter_dts(mu_dts, start, end)
    # mu_dts = [mu_dts[idx] for idx in np.random.choice(len(mu_dts), size=20, replace=False)]
    # sep_mat = get_sep_mat(extended_emg, mu_dts) # get separation matrix and test for each dimension it is working
    del_mu_dts = [mu_dts[idx] + L//2 for idx in range(len(mu_dts))]
    sep_mat = get_sep_mat_torch(extended_emg, del_mu_dts)
    muaps_idxs = [idx + Nch*jdx for idx in range(Nch) for jdx in range(L)]
    MUAPs = sep_mat[:, muaps_idxs]

    # # Get separation matrix and inverse covaraince
    # print('GET SEP MATRIX AND INVERSE COVARIANCE...')
    # extended_emg = extend_emg_torch(torch.tensor(emg.T), R)
    # inv_cov = get_inv_cov_torch(extended_emg.T, explained_var=1.0-1e-12)
    # sep_mat = get_sep_mat_torch(extended_emg, mu_dts)
    # sep_mat = torch.tensor(sep_mat @ inv_cov).to(torch.float64)
    # source_est = extended_emg @ sep_mat.T
    # spktrain = np.zeros_like(source_est[:,0].squeeze())
    # spktrain[mu_dts[0].astype(int)] = 1
    # plt.figure()
    # plt.plot(source_est[:1000, 0])
    # plt.plot(spktrain[:1000])
    # plt.show()

    # Plot MUAPs across channels
    # for mu_idx in range(MUAPs.shape[0]):
    #     ma, mi = torch.max(MUAPs[mu_idx,:]), torch.min(MUAPs[mu_idx,:])
    #     fig, ax = plt.subplots(10, 26)
    #     for idx in range(10):
    #         for jdx in range(26):
    #             ax[idx, jdx].plot(MUAPs[mu_idx, (idx + jdx)*L:(idx + jdx+1)*L])
    #             ax[idx, jdx].set_ylim([mi, ma])
    #     plt.show()
    #     plt.close(fig)
    #     # plt.savefig('MUAPs')

    # Plotting spatial distribution of MUAP energy
    for mu_idx in range(MUAPs.shape[0]):
        fig = plt.figure()
        p2p_amp = []
        var = []
        for idx in range(Nch):
            p2p_amp.append(torch.max(MUAPs[mu_idx, idx*L:(idx+1)*L]) - torch.min(MUAPs[mu_idx, idx*L:(idx+1)*L]))
            var.append(torch.var(MUAPs[mu_idx, idx*L:(idx+1)*L]))
        p2p_amp = torch.tensor(p2p_amp)
        var = torch.tensor(var)
        p2p_amp = p2p_amp[index_matrix]
        var = var[index_matrix]
        sns.heatmap(var)
        plt.show()
        plt.close(fig)
