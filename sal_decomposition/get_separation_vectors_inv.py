import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
import scipy
from sklearn.cluster import KMeans
import torch
from torch.utils.data import TensorDataset, DataLoader
from MUEdit.processing_tools import extend_emg, whiten_emg, bandpass_filter, notch_filter, get_spikes, batch_process_filters, rate_of_agreement
from get_separation_vectors_avg import rate_of_agreement2, post_process_pulses

# from load_emg_data import get_observation_matrix, get_spike_matrix, loadmat_decomp
# from MUEdit.processing_tools_old import get_spikes as get_spikes_ciara


# class LinearRegressionModel(torch.torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         # Define a single linear layer
#         self.linear = torch.torch.nn.Linear(input_dim, output_dim, bias=False)
        
#     def forward(self, x):
#         # Forward pass: pass input through the linear layer
#         return self.linear(x)

# def get_discharges(W, Y, fs):
#     '''Applies Ciara's algorithm to every source vector'''
#     print('GETTING SPIKES FROM SOURCE VECTOR...')
#     source_pred, spikes = [], []
#     for idx in tqdm(range(W.shape[0])): # for every 'feature', being a motor unit
#         spred, spike = get_spikes_ciara(W[[idx], :].T, Y, fs)
#         source_pred.append(spred)
#         spikes.append(spike)
#     source_pred, spikes = np.array(source_pred).T,spikes
#     return source_pred, spikes
def delta_pulse_trains(discharge_times, duration):
    '''Computes delta pulse trains based on sets of discharge times.'''
    pulse_trains = np.zeros((duration, len(discharge_times)), dtype=np.float32)
    for mu_idx in range(len(discharge_times)):
        pulse_trains[discharge_times[mu_idx].astype(int), mu_idx] = 1.0
    return pulse_trains

def regularized_pseudoinverse_svd(A, lambd=1e-3):
    """
    Computes the regularized pseudoinverse of matrix A using SVD-based Tikhonov regularization.
    
    Parameters:
    - A (np.array): The input matrix.
    - lambd (float): Regularization parameter (default: 1e-5).
    
    Returns:
    - A_pseudo (np.array): The regularized pseudoinverse of A.
    """
    # Perform SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Regularize the singular values
    s_reg = np.array([si / (si**2 + lambd) for si in s])
    
    # Construct the regularized pseudoinverse
    A_pseudo = Vt.T @ np.diag(s_reg) @ U.T
    return A_pseudo


def get_discharge_times(arr):
    ''' Based on discharge time format, get array of separation vectors'''
    distimes = []
    for grid_idx in range(arr['edition']['Distimeclean'].shape[0]):
        for mu_idx in range(arr[arr['edition']['Distimeclean'][grid_idx,0]].shape[0]):
            distimes.append(arr[arr[arr['edition']['Distimeclean'][grid_idx,0]][mu_idx,0]][:].ravel())
    return distimes

def open_decomp_output(DIR, name, temp_dir='./temp', MVC=20):
    ''' Open data from Simon's files so that it can be decomposed with Ciara's code.'''
    # arr = loadmat(os.path.join(DIR, name))
    with h5py.File(os.path.join(DIR, name), 'r') as arr:
        grid_idx = 0
        # Load acquisition signal
        signal = {}
        # signal['nchans'] = 64
        signal['fsamp'] = arr['signal']['fsamp'][0,0]
        signal['path'] = arr['signal']['path'][:].ravel()
        signal['target'] = arr['signal']['target'][:].ravel()
        signal['ied'] = arr['signal']['IED'][grid_idx, 0]
        # signal['electrode'] = arr['signal'][0,0][5][0, grid_idx][0]
        # signal['muscle'] = arr['signal'][0,0][6][0, grid_idx][0]
        # signal['nelectrodes'] = signal['nneedles'] + signal['ngrids']

        # Load data for all grids (right now only keep second grid)
        signal['data'] = arr['signal']['data'][:, :256]
        pulse_trains = np.hstack([ arr[arr['edition']['Pulsetrainclean'][idx,0]][:] for idx in range(4) ])
        signal['EMGmask'] = np.concatenate([arr[arr['signal']['EMGmask'][idx,0]][0,:] for idx in range(4) ])
        distimes = get_discharge_times(arr)

        decomp_dict = {} # initialising this dictionary here for later use
        mu_dict = dict(pulse_trains = None, discharge_times = [])# initialising a dictionary that is an empty nested list

    return signal, pulse_trains, distimes, decomp_dict, mu_dict


if __name__ == '__main__':
    DIR ='/home/joao/Desktop/datasets/DatasetEmanuele/Subject3'
    name = 'S_3_50_1.otb+_decomp.mat_edited.mat'
    signal, pulse_trains, distimes, decomp_dict, mu_dict = open_decomp_output(DIR, name)
    pulse_trains = delta_pulse_trains(distimes, duration=signal['data'].shape[0])
    
    # Filter EMG
    print('FILTERING EMG...')
    emg = bandpass_filter(signal['data'].T, int(signal['fsamp']))
    emg = notch_filter(emg, int(signal['fsamp']))

    # Filter out masked signals by zeroing them out
    emg[signal['EMGmask'].astype(bool), :] = 0.0

    # Extend EMG
    print('LOADING, EXTENDING, WHITENING...')
    extension_factor = int(np.round(1000/emg.shape[0]))
    extended_emg = extend_emg(np.zeros((emg.shape[0]*extension_factor, emg.shape[1] + extension_factor - 1)) , emg, extension_factor)
    # extended_emg = extended_emg[:, :extended_emg.shape[1] - (extension_factor-1)] # ensure that number of samples matches that of pulse trains
    # pulse_trains = np.vstack((pulse_trains, np.zeros((extension_factor-1, pulse_trains.shape[1])))) # extend such that temporal shape matches

    # Detrending & Whitening EMG
    extended_emg = scipy.signal.detrend(extended_emg, axis=- 1, type='constant', bp=0)
    whitened_emg, whitening_mat, dewhitening_mat = whiten_emg(extended_emg)

    # Crop only to relevant path performed
    orig_sig_size = signal['data'].shape[0]
    plateau = np.where(signal['target'] >= max(signal['target'])*0.8)[0] # finding where the plateau is        
    segment = [plateau[0],plateau[-1]]

    whitened_emg = whitened_emg[:, int(segment[0]):int(segment[1])+extension_factor]
    pulse_trains = pulse_trains.T
    pulse_trains_crop = pulse_trains[:, int(segment[0]):int(segment[1])+extension_factor]

    # Pseudoinvert the observation matrix and multiply with clean spikes to get separation vectors
    print('EMG PSEUDOINVERSE...')
    # inverted_emg = np.linalg.pinv(whitened_emg)
    inverted_emg = regularized_pseudoinverse_svd(whitened_emg)

    # Get separation vectors B
    print('EXTRACTING SEPARATION VECTORS...')
    B = pulse_trains_crop @ inverted_emg

    # Test that it actually does work...
    print('GETTING PULSE ESTIMATES BASED ON SEPARATION VECS.')
    pulse_trains_est = B @ whitened_emg    

    # Get correlations
    corrs = np.zeros((pulse_trains_crop.shape[0], pulse_trains_est.shape[0]))
    for rdx in tqdm(range(pulse_trains_crop.shape[0])):
        for cdx in range(pulse_trains_est.shape[0]):
            stat, p = scipy.stats.pearsonr(pulse_trains_crop[rdx,:], pulse_trains_est[cdx,:])
            corrs[rdx, cdx] = stat

    plt.figure()
    sns.heatmap(corrs)
    plt.title('Correlation Coefficients Between MUs')
    plt.show()

    # Get pulse trains via postprocessin
    # pulse_trains_est, discharge_times = batch_process_filters(whitened_emg, B.T,segment,1,0,orig_sig_size,int(signal['fsamp']))    pulse_trains_est, discharge_times_est = post_process_pulses(pulse_trains_est.T, signal['data'].T, plateau=segment, fsamp=signal['fsamp'], extension_factor=extension_factor, cov_thr=0.5)
    discharge_times = distimes
    pulse_trains_est, discharge_times_est = post_process_pulses(pulse_trains_est.T, emg, plateau=segment, fsamp=signal['fsamp'], extension_factor=extension_factor, cov_thr=0.5)

    corrs = np.zeros((pulse_trains.shape[0], pulse_trains_est.shape[0]))
    for rdx in tqdm(range(pulse_trains.shape[0])):
        for cdx in range(pulse_trains_est.shape[0]):
            stat, p = scipy.stats.pearsonr(pulse_trains[rdx,:], pulse_trains_est[cdx,:])
            corrs[rdx, cdx] = stat


    # Plot some ground truth vs. pred
    for idx in range(3):
        # f_est, pxx_est = scipy.signal.welch(pulse_trains_est[0,:] - pulse_trains_est[0,:].mean(), int(signal['fsamp']))
        # f, pxx = scipy.signal.welch(pulse_trains[0,:] - pulse_trains[0,:].mean(), int(signal['fsamp']))
        plt.figure()
        # plt.plot(f_est, pxx_est, color='blue')
        # plt.plot(f, pxx, color='red', linestyle='--')
        plt.plot((pulse_trains_est[idx,:] - pulse_trains_est[idx,:].mean()) / (pulse_trains_est[idx,:].std() + 1e-9), color='blue')
        plt.plot((pulse_trains[idx,:] - pulse_trains[idx,:].mean()) / (pulse_trains[idx,:].std() + 1e-9) , color='red', linestyle='--')
        plt.legend(['Estimated', 'Ground Truth'])
        plt.show()

    # roas = rate_of_agreement2(distimes, discharge_times, threshold=int(0.0005*int(signal['fsamp'])))
    # print(roas)
    roas,_ = rate_of_agreement(distimes, pulse_trains, discharge_times_est, pulse_trains_est, maxlag=40, jitter_val=0.0005, fsamp=int(signal['fsamp']), duration=pulse_trains_est.shape[1])
    print(roas)