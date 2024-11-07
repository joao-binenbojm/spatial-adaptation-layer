######################################################  EMG TENSORIZERS FOR DECOMPOSITION #######################################################################################
# from  sal_decomposition.MUEdit import emg_decomposition_final
import glob, os
import numpy as np
import pickle 
import pandas as pd
import json
from scipy.io import loadmat
import scipy
from torch.utils.data import DataLoader, TensorDataset
from networks_utils import SpatialAdaptation
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sal_decomposition.MUEdit.processing_tools import refine_mus, remove_outliers, remove_duplicates

# Module to perform spatial decomposition adaptation
class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, whiten_mat, sep_mat, extension_factor=17):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.sal = SpatialAdaptation(input_shape=grid_shape)
        self.whiten_mat = torch.nn.Linear(whiten_mat.shape[0], whiten_mat.shape[1], bias=False)
        self.sep_mat = torch.nn.Linear(sep_mat.shape[0], sep_mat.shape[1], bias=False)
        with torch.no_grad():
            self.whiten_mat.weight.copy_(whiten_mat)
            self.sep_mat.weight.copy_(sep_mat.T)
        
        self.extension_factor = extension_factor

    # def extend_emg(self, emg):
    #     '''Extend the original EMG batch given extension factor.'''
    #     extended_emg = torch.zeros((self.nchans*self.extension_factor, emg.shape[0]), requires_grad=True)
    #     for idx in range(self.extension_factor):
    #         mask = torch.ones_like(emg)
    #         mask[emg.shape[0]-idx:, :, :] = 0
    #         delayed_emg = torch.roll(emg*mask, shifts=idx, dims=0).flatten(1, 2).T # roll masked emg, hence a shift
    #         indices = idx*self.nchans + torch.arange(self.nchans)
    #         extended_emg.index_copy_(0, indices, delayed_emg) # copy delayed representation of EMG into tensor in a differentiable way.
    #     return extended_emg.T
    
    # Extend, whiten and separate sources
    def forward(self, extended_emg):
        # emg = self.extend_emg(emg)
        # Apply SAL to each slice of the extended tensor
        flat_emg = extended_emg.flatten(1, 3)
        Z = self.whiten_mat(flat_emg)
        sources = self.sep_mat(Z)
        return sources

def extend_emg_tensor(emg_grid, extension_factor=16):
    '''Extends EMG into a tensor of HxWxRxN.'''
    extended_emg = np.zeros((emg_grid.shape[0] + extension_factor - 1, extension_factor, emg_grid.shape[1], emg_grid.shape[2]))
    for idx in range(extension_factor):
        extended_emg[idx:emg_grid.shape[0]+idx,idx,:,:] = emg_grid
    return extended_emg

def get_pulse_trains(sources, plateau, N, extension_factor=17, fsamp=2048):
    '''Based on fICA source vectors, obtain the spike train from each signal.'''
    mu_count = sources.shape[1]
    pulse_trains = np.zeros([N, mu_count]) 
    discharge_times = [] # do not know size yet, so can only predefine as a list

    # Function to process spikes
    def maxk(signal, k): 
        return np.partition(signal, -k, axis=-1)[..., -k:]

    for mu_candidate in range(mu_count):
                            
        # Step 4a: 
        pulse_trains[int(plateau[0]):int(plateau[1])+ extension_factor, mu_candidate] = np.multiply(sources[:, mu_candidate],abs(sources[:, mu_candidate])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[:, mu_candidate]), distance = np.round(fsamp*0.005)+1) # peaks variable holds the indices of all peaks
        pulse_trains[:, mu_candidate] /=  np.mean(maxk(pulse_trains[:, mu_candidate], 10))
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[peaks, mu_candidate].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        discharge_times.append(peaks[np.where(kmeans.labels_ == spikes_ind)])
        print(f"Processing MU#{mu_candidate+1} out of {mu_count} MUs")

    return pulse_trains, discharge_times

def post_process_pulses(sources, emg, plateau, fsamp, extension_factor=16, cov_thr=None):
    '''After extracting pulses and discharge rates, filter them based on predefined conditions. '''

    pulse_trains, discharge_times = get_pulse_trains(sources,plateau,emg.shape[1],extension_factor=extension_factor,fsamp=fsamp)

    if np.shape(pulse_trains)[0] > 0: # if there are existing MUs
        
        # removing duplicate MUs
        # discharge_times_new, pulse_trains_new, mu_filters_new =  remove_duplicates(pulse_trains, discharge_times,discharge_times,np.squeeze(self.decomp_dict['masked_mu_filters']),np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])

        # removing outliers generating irrelvant discharge rates
        if cov_thr:
            discharge_times_new = remove_outliers(pulse_trains.T, discharge_times, fsamp, cov_thr)
            pulse_trains_new, discharge_times_new = refine_mus(emg, pulse_trains.T, discharge_times_new, fsamp)
            discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, fsamp, cov_thr)
                

    return pulse_trains_new.T, discharge_times_new


if __name__ == '__main__':
    DIR = './sal_decomposition'
    filename = 'decomposition_data.pkl'
    grid_shape = (12, 5)
    fsamp = 2048
    extension_factor = 17

    # Load initial pickle data
    with open(os.path.join(DIR, filename), 'rb') as f:
        decomp_data = pickle.load(f)
    
    # Use Dataset and DataLoader to get data
    emg_grid = decomp_data['filtered_data'].T.reshape((decomp_data['filtered_data'].shape[1],) + grid_shape) # reshape into a grid and test that it's behaving as expected
    extended_emg = extend_emg_tensor(emg_grid, extension_factor=extension_factor)
    extended_emg = torch.tensor(extended_emg, dtype=torch.float32)
    dataset = TensorDataset(extended_emg) # long EMG tensor as single data tensor
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create layer for whitening matrix and layer for separation vector matrix inside module
    whiten_mat = torch.tensor(decomp_data['whiten_mat'], dtype=torch.float32, requires_grad=True)
    sep_mat = torch.tensor(decomp_data['mu_filters'], dtype=torch.float32, requires_grad=True)

    # Create a SDA object
    sda = SpatialDecompositionAdaptation(grid_shape=grid_shape, whiten_mat=whiten_mat, sep_mat=sep_mat, extension_factor=extension_factor)

    # Collect output tensors
    output_list = []

    # Loop through the DataLoader
    sda.eval()
    with torch.no_grad():  # No gradients needed for inference
        for batch_emg in dataloader:
            # Add a channel dimension if your model expects (N, C, H, W)
            # batch_emg = batch_emg.unsqueeze(1)  # Shape becomes (batch_size, 1, H, W)

            # Forward pass through the model
            outputs = sda(batch_emg[0])  # Shape will be (batch_size, num_classes)

            # Collect outputs
            output_list.append(outputs)

    # Concatenate all outputs to form a single tensor of shape (N, num_classes)
    final_output = torch.cat(output_list, dim=0)

    plat = decomp_data['plateau']
    pulse_trains, discharge_times = post_process_pulses(final_output.detach().cpu().numpy(), decomp_data['original_data'], plateau=plat, fsamp=fsamp, extension_factor=extension_factor, cov_thr=decomp_data['parameters']['cov_thr'])
    print(final_output.shape)

    # Compare discharge times with discharge times from original code
    dt_original = decomp_data['discharge_times']
    scores = np.zeros(pulse_trains.shape[1])
    for idx in range(pulse_trains.shape[1]):
        # dts = [dt - plat[0] for dt in dt_original[idx] if (dt >= plat[0] and dt <= plat[1])]
        intersect_count = len(set(dt_original[idx]).intersection(set(discharge_times[idx])))
        scores[idx] = intersect_count / len(dt_original[idx])
        print(f'MU #{idx}, Firing Match={scores[idx]}')
    
    print(scores)

