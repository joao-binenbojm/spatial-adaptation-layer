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
from sklearn.preprocessing import StandardScaler
from sal_decomposition.MUEdit.processing_tools import refine_mus, remove_outliers, remove_duplicates
from seaborn import heatmap

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
    def __init__(self, cf_type=None):
        super(NegentropyLoss, self).__init__()
        self.cf_type = cf_type

    def forward(self, y):
        # Enforce input y is zero-mean and unit variance
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-9)
        # y = (y - y.mean()) / (y.std() + 1e-9)

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
        if self.cf_type == 'logcosh':
            negentropy = negentropy_logcosh**2
        elif self.cf_type == 'exp':
            negentropy = negentropy_exponential**2
        else:
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
        self.sep_mat = torch.nn.Linear(sep_mat.shape[1], sep_mat.shape[0], bias=False)
        with torch.no_grad():
            self.whiten_mat.weight.copy_(whiten_mat)
            self.sep_mat.weight.copy_(sep_mat)
        
        self.extension_factor = extension_factor
    
    def extend_emg(self, emg):
        '''Extend the original EMG batch given extension factor.'''
        device = emg.device
        nchans = emg.shape[1]
        extended_emg = torch.zeros((emg.shape[0] + self.extension_factor - 1, nchans*self.extension_factor)).to(device)
        for idx in range(self.extension_factor):
            extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
        return extended_emg

    # Extend, whiten and separate sources
    def forward(self, emg):
        emg_sal = self.sal(emg).squeeze()
        emg_sal = emg_sal[:, self.ycrop:emg_sal.shape[1]-self.ycrop, self.xcrop:emg_sal.shape[2]-self.xcrop] # differentiable cropping
        extended_emg = self.extend_emg(emg_sal.reshape(emg_sal.shape[0], -1))
        Z = self.whiten_mat(extended_emg)
        sources = self.sep_mat(Z)
        return sources


# def get_pulse_trains(sources, plateau, N, extension_factor=17, fsamp=2048):
#     '''Based on fICA source vectors, obtain the spike train from each signal.'''
#     mu_count = sources.shape[1]
#     pulse_trains = np.zeros([N, mu_count]) 
#     discharge_times = [] # do not know size yet, so can only predefine as a list

#     # Function to process spikes
#     def maxk(signal, k): 
#         return np.partition(signal, -k, axis=-1)[..., -k:]

#     for mu_candidate in range(mu_count):
                            
#         # Step 4a: 
#         pulse_trains[int(plateau[0]):int(plateau[1])+ extension_factor, mu_candidate] = np.multiply(sources[:, mu_candidate],abs(sources[:, mu_candidate])) # keep the negatives 
#         # Step 4b:
#         peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[:, mu_candidate]), distance = np.round(fsamp*0.005)+1) # peaks variable holds the indices of all peaks
#         pulse_trains[:, mu_candidate] /=  np.mean(maxk(pulse_trains[:, mu_candidate], 10))
#         kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[peaks, mu_candidate].reshape(-1,1)) # two classes: 1) spikes 2) noise
#         spikes_ind = np.argmax(kmeans.cluster_centers_)
#         discharge_times.append(peaks[np.where(kmeans.labels_ == spikes_ind)])
#         print(f"Processing MU#{mu_candidate+1} out of {mu_count} MUs")

#     return pulse_trains, discharge_times

# def post_process_pulses(sources, emg, plateau, fsamp, extension_factor=16, cov_thr=None):
#     '''After extracting pulses and discharge rates, filter them based on predefined conditions. '''

#     pulse_trains, discharge_times = get_pulse_trains(sources,plateau,emg.shape[1],extension_factor=extension_factor,fsamp=fsamp)

#     if np.shape(pulse_trains)[0] > 0: # if there are existing MUs
        
#         # removing duplicate MUs
#         # discharge_times_new, pulse_trains_new, mu_filters_new =  remove_duplicates(pulse_trains, discharge_times,discharge_times,np.squeeze(self.decomp_dict['masked_mu_filters']),np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])

#         # removing outliers generating irrelvant discharge rates
#         if cov_thr:
#             discharge_times_new = remove_outliers(pulse_trains.T, discharge_times, fsamp, cov_thr)
#             pulse_trains_new, discharge_times_new = refine_mus(emg, pulse_trains.T, discharge_times_new, fsamp)
#             discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, fsamp, cov_thr)
                

#     return pulse_trains_new.T, discharge_times_new


if __name__ == '__main__':
    DIR = './sal_decomposition'
    filename = 'decomposition_data.pkl'
    # grid_shape = (24, 10) 
    # grid_shape = (12, 5)
    grid_shape = (25, 10) 
    fsamp = 2048

    # Set pytorch default to float64
    torch.set_default_dtype(torch.float64)

    # Affine transformation to correct for
    xshift = 1.5 # pixel units
    yshift = 0.5 # pixel units

    # Load initial pickle data
    with open(os.path.join(DIR, filename), 'rb') as f:
        decomp_data = pickle.load(f)
    
    # Use Dataset and DataLoader to get data
    IED = 1 #decomp_data['ied']
    # scaler = StandardScaler()
    # uncropped_data = (decomp_data['uncropped_data'] - decomp_data['uncropped_data'].mean()) / (decomp_data['uncropped_data'].std() + 1e-12) 
    uncropped_data = decomp_data['uncropped_data']
    ycrop,xcrop = decomp_data['parameters']['ycrop'], decomp_data['parameters']['xcrop']
    nchans = (grid_shape[0]-2*abs(ycrop)) * (grid_shape[1]-2*abs(xcrop))
    emg_grid = uncropped_data.T.reshape((uncropped_data.shape[1], 1) + grid_shape)# reshape into a grid and test that it's behaving as expected
    emg_grid = torch.tensor(emg_grid, dtype=torch.float64, requires_grad=True)
    emg_grid = emg_grid + torch.randn_like(emg_grid) *0.1*emg_grid.std()
    # extension_factor = int(np.round(1000/nchans))  #17 #33 # 25 # 33 # 17
    extension_factor = decomp_data['parameters']['ext_factor']

    # extended_emg = extend_emg_tensor(emg_grid, extension_factor=extension_factor)
    # extended_emg = torch.tensor(extended_emg, dtype=torch.float32)

    # Create layer for whitening matrix and layer for separation vector matrix inside module
    whiten_mat = torch.tensor(decomp_data['whiten_mat'], requires_grad=True)
    # sep_mat = torch.tensor(np.delete(decomp_data['mu_filters'], 14, axis=1), dtype=torch.float32, requires_grad=True)
    sep_mat = torch.tensor(decomp_data['mu_filters'], requires_grad=True)
    ica_loss = NegentropyLoss() #KurtosisLoss() #NegentropyLoss() # loss function for updating SAL parameters
    device = 'cuda' # 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    # Define the grid range and density
    num_points = 20 # e.g., 100 points per unit

    # Getting torch meshgrid
    x = torch.linspace(-xcrop, xcrop, num_points) #.round(decimals=3)
    y = torch.linspace(-ycrop, ycrop, num_points) #.round(decimals=3)

    # Step 5: Apply the shifts (this will be done for all xi and yi values at once)
    sda = SpatialDecompositionAdaptation(grid_shape=grid_shape, whiten_mat=whiten_mat, sep_mat=sep_mat, xcrop=xcrop, ycrop=ycrop, extension_factor=extension_factor).to(device)
    with torch.no_grad():
        base_outputs = sda(emg_grid.to(device))
        base_loss = ica_loss(base_outputs).item()

    plt.figure()
    plt.plot(base_outputs.detach().cpu().numpy()[:, 0])
    plt.show()

    loss_arr = torch.zeros(y.shape[0], x.shape[0])

    # Initialize module
    sal_test = SpatialAdaptation(input_shape=grid_shape).eval()
    for param in sda.parameters():
        param.requires_grad = False

    # Sample transform
    with torch.no_grad():
        for param in sal_test.parameters():
            param.requires_grad = False
        sal_test.yshift.copy_(torch.tensor(2*yshift/emg_grid.shape[2]))
        sal_test.xshift.copy_(torch.tensor(2*xshift/emg_grid.shape[3]))
        emg_grid = sal_test(emg_grid) # compute this to see if translation matches expected

    # Sample parameters
    with torch.no_grad():
        for xidx, xi in enumerate(tqdm(x)):
            for yidx, yi in enumerate(y):
                emg_grid_test = emg_grid.clone()
                sda.sal.yshift.copy_(torch.tensor(2*yi/emg_grid.shape[2]).to(device))
                sda.sal.xshift.copy_(torch.tensor(2*xi/emg_grid.shape[3]).to(device))
                outputs = sda(emg_grid_test.to(device))
                loss = ica_loss(outputs)
                loss_arr[yidx, xidx] = loss.item()

    plt.figure()
    plt.title(f'Kurtosis Loss Landscape (Post (y={yshift*IED}m, x={xshift*IED}m) translation)')
    ax = heatmap(np.array(loss_arr)/base_loss, xticklabels=np.around((x*IED).tolist(), 3), yticklabels=np.around((y*IED).tolist(), 3))
    ax.set(xlabel='Circumferential Shifts (m)', ylabel='Longitudinal Shifts (m)')
    ax.text(np.where(np.array(x)>=-xshift)[0][0] + 0.5, np.where(y>=-yshift)[0][0]+0.5, 'X', color='green', ha='center', va='center', fontsize=16)
    # ax.text(0.5,0.5,'X',color='green',ha='center',va='center',fontsize=32)
    # plt.show()
    plt.savefig('loss.jpg')
    print()

    # Determine minimum location in loss landscape and check outputs at that location
    loss_min = loss_arr.min()
    yopt, xopt = torch.where(loss_arr == loss_min) # get optional shifts and apply them to the original grid
    print(f'loss_min: {loss_min}, yopt: {y[yopt]}, xopt: {x[xopt]}')

    # Apply 'optimal' transform and see if we still get spikes!
    yshiftopt, xshiftopt = y[yopt].item(), x[xopt].item()
    with torch.no_grad():
        sda.sal.yshift.copy_(torch.tensor(2*yshiftopt/emg_grid.shape[2]).to(device))
        sda.sal.xshift.copy_(torch.tensor(2*xshiftopt/emg_grid.shape[3]).to(device))
        outputs = sda(emg_grid.to(device))
        loss = ica_loss(outputs).item()
    
    spts = outputs.detach().cpu().numpy()
    plt.figure()
    plt.plot(spts[:, 0] / spts[:, 0].max())
    plt.plot(base_outputs.detach().cpu().numpy()[:, 0] / base_outputs.detach().cpu().numpy()[:, 0].max())
    plt.legend(['Estimated Spikes', 'Original Spikes'])
    plt.show()
    print()
    


