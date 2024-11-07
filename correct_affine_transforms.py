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
        mean_Y = Y.mean(dim=0, keepdim=True)
        centered_Y = Y - mean_Y

        # Calculate kurtosis for each component
        # Fourth moment: E[Y_i^4]
        fourth_moment = torch.mean(centered_Y ** 4, dim=0)
        
        # Second moment (variance): E[Y_i^2]
        second_moment = torch.mean(centered_Y ** 2, dim=0)
        
        # Kurtosis for each component: (E[Y_i^4] / (E[Y_i^2])^2) - 3
        kurtosis = fourth_moment / (second_moment ** 2) - 3
        
        # Loss as negative absolute kurtosis to maximize independence
        loss = -torch.sum(torch.abs(kurtosis))
        
        return loss
    

class NegentropyLoss(torch.nn.Module):
    def __init__(self):
        super(NegentropyLoss, self).__init__()

    def forward(self, y):
        # Assume input y is zero-mean and unit variance

        # Log-cosh contrast function
        G_y_logcosh = torch.log(torch.cosh(y))
        G_v_logcosh = torch.log(torch.cosh(torch.randn_like(y)))
        
        # Exponential contrast function
        G_y_exponential = -torch.exp(-y**2 / 2)
        G_v_exponential = -torch.exp(-torch.randn_like(y)**2 / 2)
        
        # Combine both contrast functions for negentropy
        negentropy_logcosh = torch.mean(G_y_logcosh) - torch.mean(G_v_logcosh)
        negentropy_exponential = torch.mean(G_y_exponential) - torch.mean(G_v_exponential)
        
        # Sum both to form the final combined negentropy
        negentropy = negentropy_logcosh + negentropy_exponential
        
        # Return the negative to make this a loss function (minimize -J(y))
        return -negentropy

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
    
    # Extend, whiten and separate sources
    def forward(self, extended_emg):
        # N, R, H, W = extended_emg.shape
        # extended_emg = extended_emg.reshape(N*R, 1, H, W)
        emg_sal = self.sal.forward(extended_emg)
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
    grid_shape = (24, 10)
    fsamp = 2048
    extension_factor = int(1000/nchans) #17

    # Affine transformation to correct for
    rot_angle = 0
    xshift = 1
    yshift = 1

    # Load initial pickle data
    with open(os.path.join(DIR, filename), 'rb') as f:
        decomp_data = pickle.load(f)
    
    # Use Dataset and DataLoader to get data
    emg_grid = decomp_data['filtered_data'].T.reshape((decomp_data['filtered_data'].shape[1],) + grid_shape) # reshape into a grid and test that it's behaving as expected

    extended_emg = extend_emg_tensor(emg_grid, extension_factor=extension_factor)
    extended_emg = torch.tensor(extended_emg, dtype=torch.float32)

    # Applying virtual shift
    sal_test = SpatialAdaptation(input_shape=grid_shape).eval()
    with torch.no_grad():
        sal_test.yshift.copy_(torch.tensor(yshift))
    for param in sal_test.parameters():
        param.requires_grad = False
        extended_emg_transform = sal_test(extended_emg) # compute this to see if translation matches expected

    rms_transform = torch.sqrt(torch.tensor(extended_emg_transform**2).mean(dim=(0,1)))
    rms = torch.sqrt(torch.tensor(extended_emg**2).mean(dim=(0,1)))
    # plt.figure()
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(np.array(rms))
    plt.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(np.array(rms_transform))
    plt.colorbar(im2, ax=axs[1])
    plt.show()


    dataset = TensorDataset(extended_emg_transform) # long EMG tensor as single data tensor
    dataloader = DataLoader(dataset, batch_size=extended_emg.shape[0], shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    # Create layer for whitening matrix and layer for separation vector matrix inside module
    
    whiten_mat = torch.tensor(decomp_data['whiten_mat'], dtype=torch.float32, requires_grad=True)
    sep_mat = torch.tensor(decomp_data['mu_filters'], dtype=torch.float32, requires_grad=True)

    # Create a SDA object
    nepochs = 100
    sda = SpatialDecompositionAdaptation(grid_shape=grid_shape, whiten_mat=whiten_mat, sep_mat=sep_mat, extension_factor=extension_factor).to(device)
    ica_loss = KurtosisLoss() # loss function for updating SAL parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sda.parameters()),
                                                lr=1e-2, weight_decay=0)

    # Freeze all parameters excep for sal
    for param in sda.parameters():
        param.requires_grad = False

    for param in sda.sal.parameters():
        param.requires_grad = True

    # Collect output tensors
    output_list = []
    losses = []
    xshifts,yshifts,angles = [], [], []

    # Loop through the DataLoader
    for ne in tqdm(range(nepochs)):
        for batch_emg in dataloader:
            # Forward pass through the model
            outputs = sda(batch_emg[0].to(device)).to(device)  # Shape will be (batch_size, num_classes)

            # Compute ICA Loss and backprop    
            loss = ica_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect outputs and loss
            output_list.append(outputs)
            losses.append(loss.item())
            xshifts.append(sda.sal.xshift.item())
            yshifts.append(sda.sal.yshift.item())
            angles.append(sda.sal.rot_theta.item())

    # Concatenate all outputs to form a single tensor of shape (N, num_classes)
    final_output = torch.cat(output_list, dim=0)
    plt.figure()
    plt.plot(np.array(losses).ravel())
    plt.show()
    print()

    plt.figure()
    # plt.plot(np.array(xshifts).ravel(), label='xshift-pred')
    # plt.plot(np.array(yshifts).ravel(), label='yshift-pred')
    # plt.plot(np.array(angles).ravel(), label='angle-pred')
    plt.plot(grid_shape[1]*(np.array(xshifts).ravel())/2)
    plt.plot(grid_shape[0]*(np.array(yshifts).ravel())/2)
    plt.hlines(y=[xshift, yshift], xmin=0, xmax=len(np.array(xshifts).ravel()-1), linestyles='dashed', label='ground truth')
    plt.legend(['xshift-pred','yshift-pred'])
    plt.show()

    # plat = decomp_data['plateau']
    # pulse_trains, discharge_times = post_process_pulses(final_output.detach().cpu().numpy(), decomp_data['original_data'], plateau=plat, fsamp=fsamp, extension_factor=extension_factor, cov_thr=decomp_data['parameters']['cov_thr'])
    # print(final_output.shape)

    # # Compare discharge times with discharge times from original code
    # dt_original = decomp_data['discharge_times']
    # scores = np.zeros(pulse_trains.shape[1])
    # for idx in range(pulse_trains.shape[1]):
    #     # dts = [dt - plat[0] for dt in dt_original[idx] if (dt >= plat[0] and dt <= plat[1])]
    #     intersect_count = len(set(dt_original[idx]).intersection(set(discharge_times[idx])))
    #     scores[idx] = intersect_count / len(dt_original[idx])
    #     print(f'MU #{idx}, Firing Match={scores[idx]}')
    
    # print(scores)

