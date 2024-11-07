######################################################  EMG TENSORIZERS FOR DECOMPOSITION #######################################################################################
# from  sal_decomposition.MUEdit import emg_decomposition_final
import glob, os
import numpy as np
import pickle 
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import scipy
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode 
from networks_utils import SpatialAdaptation
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sal_decomposition.MUEdit.processing_tools import refine_mus, remove_outliers, remove_duplicates

class KurtosisLoss(torch.nn.Module):
    def __init__(self):
        super(KurtosisLoss, self).__init__()

    def forward(self, Y):
        # Y is assumed to have shape (batch_size, num_components)
        
        # Calculate the mean and variance of each component along the batch dimension
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-9)
        # Y = Y - Y.mean(dim=0, keepdim=True)

        # Calculate kurtosis for each component
        # Fourth moment: E[Y_i^4]
        fourth_moment = torch.mean(Y ** 4, dim=0)
        
        # Second moment (variance): E[Y_i^2]
        second_moment = torch.mean(Y ** 2, dim=0)
        
        # Kurtosis for each component: (E[Y_i^4] / (E[Y_i^2])^2) - 3
        kurtosis = fourth_moment / (second_moment ** 2) - 3
        
        # Loss as negative absolute kurtosis to maximize independence
        loss = -torch.sum(torch.abs(kurtosis))
        
        return loss

class NegentropyLoss(torch.nn.Module):
    def __init__(self):
        super(NegentropyLoss, self).__init__()

    def forward(self, y):
        # Enforce input y is zero-mean and unit variance
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-9)
        
        # Log-cosh contrast function
        G_y_logcosh = torch.log(torch.cosh(y))
        G_v_logcosh = torch.log(torch.cosh(torch.randn_like(y)))

        # Square contrast function
        # negentropy = torch.mean(torch.square(y)) - torch.log(torch.cosh(torch.randn_like(y)))
        
        # Exponential contrast function
        G_y_exponential = -torch.exp(-y**2 / 2)
        G_v_exponential = -torch.exp(-torch.randn_like(y)**2 / 2)
        
        # Combine both contrast functions for negentropy
        negentropy_logcosh = torch.mean(G_y_logcosh) - torch.mean(G_v_logcosh)
        negentropy_exponential = torch.mean(G_y_exponential) - torch.mean(G_v_exponential)
        
        # Sum both to form the final combined negentropy
        negentropy = negentropy_logcosh**2 + negentropy_exponential**2
        
        # Return the negative to make this a loss function (minimize -J(y))
        return -negentropy
    
# # Module to perform spatial decomposition adaptation
# class SpatialDecompositionAdaptation(torch.nn.Module):    
#     # build the constructor
#     def __init__(self, grid_shape, whiten_mat, sep_mat, extension_factor=17):
#         super(SpatialDecompositionAdaptation, self).__init__()
#         self.grid_shape = grid_shape
#         self.nchans = torch.prod(torch.tensor(grid_shape))
#         self.sal = SpatialAdaptation(input_shape=grid_shape)

#         self.whiten_mat = torch.nn.Linear(whiten_mat.shape[0], whiten_mat.shape[1], bias=False)
#         self.sep_mat = torch.nn.Linear(sep_mat.shape[0], sep_mat.shape[1], bias=False)
#         with torch.no_grad():
#             self.whiten_mat.weight.copy_(whiten_mat)
#             self.sep_mat.weight.copy_(sep_mat.T)
        
#         self.extension_factor = extension_factor
    
#     # Extend, whiten and separate sources
#     def forward(self, extended_emg):
#         # N, R, H, W = extended_emg.shape
#         # extended_emg = extended_emg.reshape(N*R, 1, H, W)
#         emg_sal = self.sal.forward(extended_emg)
#         flat_emg = emg_sal.flatten(1, 3)
#         Z = self.whiten_mat(flat_emg)
#         sources = self.sep_mat(Z)
#         return sources

# Module to perform spatial decomposition adaptation
class SpatialDecompositionAdaptation(torch.nn.Module):    
    # build the constructor
    def __init__(self, grid_shape, whiten_mat, sep_mat, ycrop, xcrop, extension_factor=17):
        super(SpatialDecompositionAdaptation, self).__init__()
        self.grid_shape = grid_shape
        self.nchans = torch.prod(torch.tensor(grid_shape))
        self.sal = SpatialAdaptation(input_shape=grid_shape)
        self.ycrop = ycrop
        self.xcrop = xcrop

        self.whiten_mat = torch.nn.Linear(whiten_mat.shape[0], whiten_mat.shape[1], bias=False)
        self.sep_mat = torch.nn.Linear(sep_mat.shape[0], sep_mat.shape[1], bias=False)
        with torch.no_grad():
            self.whiten_mat.weight.copy_(whiten_mat)
            self.sep_mat.weight.copy_(sep_mat.T)
        
        self.extension_factor = extension_factor
    
    # Extend, whiten and separate sources
    def forward(self, extended_emg):
        # N, R, H, W = extended_emg.shape
        # extended_emg = extended_emg.reshape(N*R, 1, H, W)
        emg_sal = self.sal.forward(extended_emg)
        emg_sal = emg_sal[:, :, self.ycrop:emg_sal.shape[2]-self.ycrop, self.xcrop:emg_sal.shape[3]-self.xcrop] # differentiable cropping!
        flat_emg = emg_sal.flatten(1, 3)
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
    extension_factor = 25 # 33 # 17

    # Affine transformation to correct for
    xshift = 1 # pixel units
    yshift = 1 # pixel units

    # Define the grid range and density
    density = 50  # e.g., 100 points per unit
    # grid_min, grid_max = -1.0, 1.0

    # Calculate the number of points based on density
    num_points = density #int((grid_max - grid_min) * density) + 1

    # Generate the grid in each dimension
    # x = torch.linspace(-xcrop, xcrop, num_points).round(decimals=3)
    # x = torch.zeros(1)
    # y = torch.linspace(-ycrop, ycrop, num_points).round(decimals=3)

    # # Create a meshgrid
    # xx, yy = torch.meshgrid(x, y, indexing='ij')
    # grid = torch.stack([xx, yy], dim=-1)
    # loss_arr = torch.zeros(y.shape[0], x.shape[0])    

    # Sample trasnform

    test_img = torch.zeros(1,1,10,10)
    test_img[0, 0, 1:9, 1:9] = 1

    plt.figure()
    plt.imshow(test_img.squeeze().numpy())
    plt.savefig('test.jpg')

    sal_test = SpatialAdaptation(input_shape=test_img.shape[2:]).eval()
    with torch.no_grad():
        sal_test.yshift.copy_(torch.tensor(2*yshift/test_img.shape[2]))
        sal_test.xshift.copy_(torch.tensor(2*xshift/test_img.shape[3]))
        for param in sal_test.parameters():
            param.requires_grad = False
        test_img= sal_test(test_img) # compute this to see if translation matches expected
    plt.figure()
    plt.imshow(test_img.squeeze().numpy())
    plt.savefig('test_sal.jpg')
    print()

    # for param in sda.parameters():
    #     param.requires_grad = False
    # with torch.no_grad():
    #     for xidx, xi in enumerate(tqdm(x)):
    #         for yidx, yi in enumerate(y):
    #             sda.sal.yshift.copy_(torch.tensor(yi).to(device))
    #             sda.sal.xshift.copy_(torch.tensor(xi).to(device))
    #             outputs = sda(extended_emg.to(device))
    #             loss = ica_loss(outputs)
    #             loss_arr[yidx, xidx] = loss.item()

    # from seaborn import heatmap
    # plt.figure()
    # ax = heatmap(np.array(loss_arr)/base_loss.item(), xticklabels=np.around(x.tolist(), 2), yticklabels=np.around(y.tolist(), 2))
    # # ax.text(np.where(np.array(x)>=-xshift)[0][0], np.where(y>=-yshift)[0][0], 'X', color='green', ha='center', va='center', fontsize=32)
    # ax.text(0.5,0.5,'X',color='green',ha='center',va='center',fontsize=32)
    # # plt.show()
    # plt.savefig('loss.jpg')
    # print()