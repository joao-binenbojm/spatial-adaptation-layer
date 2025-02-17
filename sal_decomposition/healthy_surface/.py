import numpy as np
import pandas as pd
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
from scipy.optimize import linear_sum_assignment


def apply_affine(emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1):
    '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''

    N, C, H, W = emg_grid.shape
    Tx, Ty = torch.tensor(2*Tx/W), torch.tensor(2*Ty/H) # Normalize translation values automatically
    theta, xscale, yscale = torch.tensor(theta) / torch.pi, torch.tensor(xscale), torch.tensor(yscale)

    T = torch.cat([ # Translation Matrix
        torch.stack([torch.tensor(1.0), torch.tensor(0.0), Tx]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(1.0), Ty]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)
    R = torch.cat([ # Rotation Matrix
        torch.stack([torch.cos(theta), -torch.sin(theta), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.sin(theta), torch.cos(theta), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)
    Sc = torch.cat([ # Scaling Matrix
        torch.stack([xscale, torch.tensor(0.0), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), yscale, torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)

    # theta = Sc @ R @ T # learning order
    theta = T @ R @ Sc
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(N,1,1)
    grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=False)
    xresamp = torch.nn.functional.grid_sample(emg_grid, grid)
    
    return xresamp

# def frequency_regularity_loss(spike_train, fs, target_freq=20.0, sigma=10.0):
#     # Standardize spike train (ensures larger MUAPs are not prioritized)
#     spike_train = (spike_train - spike_train.mean(dim=0, keepdim=True)) / spike_train.std(dim=0, keepdim=True)

#     # Compute power spectrum
#     spectrum = torch.fft.rfft(spike_train, dim=0)
#     freqs = torch.fft.rfftfreq(spike_train.shape[0], d=1/fs)
    
#     # Create gaussian window around expected frequency
#     window = torch.exp(-(freqs - target_freq)**2 / (2 * sigma**2)).unsqueeze(1).to(spike_train.device)
    
#     # Penalize power outside the expected frequency band
#     spectral_penalty = -torch.sum(torch.abs(spectrum) * window)
    
#     return spectral_penalty

def get_base_loss(emg_grid_transform, sda, batch_size=2048, loss='kurtosis', device='cuda'):
    '''Compute the base loss for the given emg_grid_transform and sda.'''
    # Split data into batches and compute average loss
    n_batches = emg_grid_transform.shape[0] // batch_size
    total_loss = 0
    
    # Choose loss function
    if loss == 'kurtosis':
        ica_loss = KurtosisLoss()
    else:
        ica_loss = NegentropyLoss()

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = emg_grid_transform[start_idx:end_idx]
            
            # Get source estimates for this batch
            source_est = sda(batch.to(device))
            cur_loss = ica_loss(source_est).item()
            # freq_loss = frequency_regularity_loss(source_est, fs, target_freq=20, sigma=10.0)
            # cur_loss = cur_loss + freq_reg*freq_loss
            total_loss += cur_loss
        
    # Average loss across batches
    avg_loss = total_loss / n_batches    
    return avg_loss

def search_fit_sda(emg_grid_transform, sda, base_loss=1.00, npoints=50, nepochs=50, batch_size=2048, lr=1e-4, boundaries=(2.5, 2.5, 10*np.pi/180), device='cpu', loss='kurtosis', R=16, frozen_sep_mat=True, plot=1):
    ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''
    
    N, C, H, W = emg_grid_transform.shape
    if loss == 'kurtosis':
        ica_loss = KurtosisLoss()
    else:
        ica_loss = NegentropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sda.parameters()),
                                                lr=lr)                 

    # Collect output tensors
    output_list = []
    losses = []
    xshifts,yshifts,angles,xscales,yscales = [], [], [], [], []

    # Freeze all parameters except for SAL parameters
    for param in sda.parameters():
        param.requires_grad = False
    
    # Searching through initial conditions
    print('SAMPLING AND EVALUATING INITIAL CONDITIONS...')
    losses = torch.zeros(npoints)
    engine = scipy.stats.qmc.LatinHypercube(d=5)
    engine = scipy.stats.qmc.LatinHypercube(d=3)
    init_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float64)-1 # scale from [0,1] to [-1, 1]
    init_params[:,0], init_params[:,1], init_params[:, 2] = 2*boundaries[0]*init_params[:,0]/W, 2*boundaries[1]*init_params[:,1]/H, boundaries[2]*init_params[:, 2]/np.pi
    # init_params[:, 3] = torch.pow((1 + torch.abs(init_params[:, 3])*boundaries[3]), torch.sign(init_params[:, 3]) ) # generates scalings appropriately
    # init_params[:, 4] = torch.pow((1 + torch.abs(init_params[:, 4])*boundaries[4]), torch.sign(init_params[:, 4]) )

    init_params = init_params.to(device)
    sda.train() # leave batch norm parameters adaptive
    with torch.no_grad():
        
        for npoint in tqdm(range(npoints)):
            # Set initial conditions
            sda.sal.xshift.data, sda.sal.yshift.data, sda.sal.rot_theta.data = init_params[npoint, :3]
            # sda.sal.xscale.data, sda.sal.yscale.data = init_params[npoint, 3:]

            # Evaluate loss function at given condition across batches
            n_batches = emg_grid_transform.shape[0] // batch_size
            total_loss = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, emg_grid_transform.shape[0])
                batch = emg_grid_transform[start_idx:end_idx]
                
                # Get source estimates for this batch
                source_est = sda(batch.to(device))
                total_loss += ica_loss(source_est).item()
            
            # Average loss across batches
            avg_loss = total_loss / n_batches
            losses[npoint] = avg_loss

        if npoints > 0:
            losses = losses / base_loss # normalize by baseline loss
            sda.sal.xshift.data, sda.sal.yshift.data, sda.sal.rot_theta.data = init_params[losses.argmax(), :3] # get best initialization
            # sda.sal.xscale.data, sda.sal.yscale.data = init_params[losses.argmax(), 3:]
            print(f'TOP 5 LOSS VALUES SAMPLED: {torch.topk(losses, k=torch.min(torch.tensor([npoints, 5])))}')

    # Make SAL parameters learnable
    # for param in sda.sal.parameters():
    if frozen_sep_mat:
        for param in sda.sal.parameters():
            param.requires_grad = True        
    else:
        for param in sda.parameters():
            param.requires_grad = True

    # Loop through the DataLoader
    print('TRAINING FROM BEST INIT. CONDITION...')
    losses = []
    for ne in tqdm(range(nepochs)):
        # Forward pass through the model
        n_batches = emg_grid_transform.shape[0] // batch_size
        epoch_loss = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, emg_grid_transform.shape[0])
            batch = emg_grid_transform[start_idx:end_idx]
            
            # Get source estimates for this batch
            source_est = sda(batch.to(device))
            batch_loss = ica_loss(source_est)
            
            # Backprop for this batch
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        # Average loss for the epoch
        epoch_loss = epoch_loss / n_batches

        print('LOSS:', epoch_loss/base_loss)
        optimizer.step()
        print(f'PARAMS:\n xshift: {W*sda.sal.xshift.item()/2}, yshift: {H*sda.sal.yshift.item()/2}, theta: {sda.sal.rot_theta.item()} ')
        # print(f'xscale: {sda.sal.xscale.item()}, yscale: {sda.sal.yscale.item()}')
        # Collect outputs and loss
        losses.append(epoch_loss)
        xshifts.append(sda.sal.xshift.item())
        yshifts.append(sda.sal.yshift.item())
        angles.append(sda.sal.rot_theta.item())
        # xscales.append(sda.sal.xscale.item())
        # yscales.append(sda.sal.yscale.item())

    # Get final outputs, i.e. optimal souces
    with torch.no_grad():
        # Process in batches and concatenate results
        batch_outputs = []
        total_samples = emg_grid_transform.shape[0]
        
        # Process full batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch = emg_grid_transform[start_idx:end_idx]
            
            # Get outputs for this batch
            batch_output = sda(batch.to(device))
            # Move to CPU before appending to save GPU memory
            batch_outputs.append(batch_output.cpu())
                        
        # Concatenate all batches
        final_outputs = torch.cat(batch_outputs, dim=0).to(device)

    sources = final_outputs.detach().cpu()
    return sda, sources, losses

def loss_sampling(emg_grid_transform, sda, base_loss=1.0, T=(0.0, 0.0), bounds=(0.0, 0.0), batch_size=2048, num_points=20, loss='kurtosis', device='cpu'):
    ''' Method used to sample the loss landscape.'''
    N, C, H, W = emg_grid_transform.shape
    if loss == 'kurtosis':
        ica_loss = KurtosisLoss()
    else:
        ica_loss = NegentropyLoss()

    # Getting torch meshgrid
    Tx, Ty = T
    xbounds, ybounds = bounds
    x = torch.linspace(-torch.tensor(xbounds), torch.tensor(xbounds), num_points)
    y = torch.linspace(-torch.tensor(ybounds), torch.tensor(ybounds), num_points)
    loss_arr = torch.zeros(y.shape[0], x.shape[0])

    # Sample parameters
    sda.train() # leave layer norm adaptive and running
    with torch.no_grad():
        for xidx, xi in enumerate(tqdm(x)):
            for yidx, yi in enumerate(y):
                emg_grid_test = emg_grid_transform.clone().detach()
                sda.sal.yshift.copy_(torch.tensor(2*yi/H).to(device))
                sda.sal.xshift.copy_(torch.tensor(2*xi/W).to(device))

                # Process in batches and compute average loss
                total_samples = emg_grid_test.shape[0]
                batch_losses = []
                
                # Process full batches
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch = emg_grid_test[start_idx:end_idx]
                    
                    # Get outputs and loss for this batch
                    batch_output = sda(batch.to(device))
                    batch_loss = ica_loss(batch_output)
                    # freq_loss = frequency_regularity_loss(batch_output, fs, target_freq=20, sigma=10.0)
                    # batch_loss = batch_loss + freq_reg*freq_loss
                    batch_losses.append(batch_loss.item())
                            
                # Average the losses across all batches
                loss = sum(batch_losses) / len(batch_losses)
                loss_arr[yidx, xidx] = loss
    
    plt.figure()
    # plt.title(f'Kurtosis Loss Landscape (Post (y={Ty}, x={Tx}) translation)')
    # ax = sns.heatmap(np.array(loss_arr)/self.base_loss, xticklabels=np.around((x).tolist(), 3), yticklabels=np.around((y).tolist(), 3))
    ax = sns.heatmap(np.array(loss_arr)/base_loss)
    # ax.set_xticks([], [])
    # ax.set_yticks([], [])
    ax.set(xlabel='Circumferential Shifts (mm)', ylabel='Longitudinal Shifts (mm)')
    ax.text(np.where(np.array(x)>=-Tx)[0][0] + 0.5, np.where(y>=-Ty)[0][0]+0.5, 'X', color='green', ha='center', va='center', fontsize=16)
    
    plt.savefig('loss_landscape.jpg')
    print()

    return loss_arr

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

def get_inv_cov(signal, explained_var=0.99):

    """ Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization. """
    cov_mat = np.cov(np.squeeze(signal),bias=True)
    print('FINISHED GETTING COVARIANCE MATRIX...')
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eigh(cov_mat)
    print('FINISHED GETTING EIGENDECOMPOSITION...')
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)

    # sorted_evalues = np.sort(evalues)[::-1]
    sorted_idxs = np.argsort(evalues)[::-1] # sort in descending order
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    cum_explained_var = evalues.cumsum() / evalues.sum()
    evalues, evectors = evalues[cum_explained_var <= explained_var], evectors[:, cum_explained_var <= explained_var]

    inv_cov = evectors @ np.diag(1 / (evalues)) @ np.transpose(evectors)
    return inv_cov

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

def get_inv_cov_tikhonov(signal, reg=1e-6):
    ''' Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization all in PyTorch (compute in torch.float64 to avoid precision issues). '''
    cov_mat = torch.cov(signal.to(torch.float64))
    cov_mat = cov_mat + reg*torch.eye(cov_mat.shape[0]).to(torch.float64)
    evalues, evectors  = torch.linalg.eigh(cov_mat)
    sorted_idxs = torch.argsort(evalues, descending=True)
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
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

def get_target_boundaries(target, threshold=0.9):
    '''Takes in signal and edition dictionaries and returns the data and edition dictionaries with the target timestamps only.'''
    target_max = np.max(target)
    # Find rising and falling edges of target signal
    rising_edge = np.where(target >= threshold*target_max)[0][0] + 5000
    falling_edge = np.where(target >= threshold*target_max)[0][-1] - 5000
 
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

def make_grid(emg, index_matrix):
    '''Function that converts EMG grid into the dimensions of a batch of images, expected by the affine transforms and decomposition module. (Input shape H, W, T)'''
    emg_grid = torch.tensor(emg[index_matrix, :]).to(torch.float64)
    emg_grid = emg_grid.permute(2, 0, 1).unsqueeze(1)
    return emg_grid

def get_sep_mat(extended_emg, dts):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix'''
    N = len(dts) # number of MUs
    sep_mat = np.zeros((N, extended_emg.shape[0]))
    for idx in range(N):
        sep_mat[idx, :] = (extended_emg[:, dts[idx].astype(int)]).mean(axis=1)
    return sep_mat

def get_sep_mat_torch(extended_emg, dts):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    N = len(dts) # number of MUs
    sep_mat = torch.zeros((N, extended_emg.shape[0])).to(torch.float64)
    for idx in range(N):
        sep_mat[idx, :] = (extended_emg[:, dts[idx].astype(int)]).mean(dim=1)
    return sep_mat

def get_sep_mat_pseudo_inv(extended_emg, dts, rcond=1e-3):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    N = len(dts) # number of MUs
    y_inv = torch.linalg.pinv(extended_emg, rcond=rcond)
    spike_trains = torch.zeros((N, extended_emg.shape[1])).to(torch.float64)
    for idx in range(N):
        spike_trains[idx, dts[idx]] = 1.0
    sep_mat = spike_trains @ y_inv
    return sep_mat

def kurt_filt_sources(Y):
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
    filt_kurt = kurtosis > kurtosis.median()
    # _, filt_kurt = torch.topk(kurtosis, 1)
    
    return filt_kurt

def extend_emg_torch(emg, R):
    '''Extend the original EMG batch given extension factor.'''
    device = emg.device
    nchans = emg.shape[1]
    extended_emg = torch.zeros((emg.shape[0] + R - 1, nchans*R)).to(torch.float64).to(device)
    for idx in range(R):
        extended_emg[idx:emg.shape[0]+idx, idx*nchans:(idx+1)*nchans] = emg
    return extended_emg

def get_silohuette( sources_pred, distance=4):
    '''Get silhouette values given source predictions.'''
    
    # Step 4b:
    sils = np.zeros(sources_pred.shape[1])
    pred_dts = []
    for mu_idx in range(sources_pred.shape[1]):
        source_pred = sources_pred[:, mu_idx] # get a single source prediction
        source_pred = np.multiply(source_pred, source_pred) # get squared sources
        peaks, _ = scipy.signal.find_peaks(source_pred.squeeze(), distance=distance) # default about 2ms 
        source_pred /=  np.mean(maxk(source_pred[peaks], 10))
        if len(peaks) > 1:
            kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            # indices of the spike and noise clusters (the spike cluster should have a larger value)
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            noise_ind = np.argmin(kmeans.cluster_centers_)
            # get the points that correspond to each of these clusters
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            noise = peaks[np.where(kmeans.labels_ == noise_ind)]
            # calculate the centroids
            spikes_centroid = kmeans.cluster_centers_[spikes_ind]
            noise_centroid = kmeans.cluster_centers_[noise_ind]
            # difference between the within-cluster sums of point-to-centroid distances 
            intra_sums = (((source_pred[spikes]- spikes_centroid)**2).sum()) 
            # difference between the between-cluster sums of point-to-centroid distances
            inter_sums = (((source_pred[spikes] - noise_centroid)**2).sum())
            sil = (inter_sums - intra_sums) / max(intra_sums, inter_sums)  
        else:
            sil = 0
        sils[mu_idx] = sil
        pred_dts.append(spikes)
    return pred_dts, sils

def spike_scores(dts, dts_pred):
    ''' For each motor unit, compute the spiking accuracy, sensitivity and precision.'''
    scores = {'sensitivity': np.zeros(len(dts)), 'precision': np.zeros(len(dts))}
    for mu_idx in range(len(dts)):
        gt, pred = set(dts[mu_idx].tolist()), set(dts_pred[mu_idx]) # account for delay induced
        tps = len(gt.intersection(pred)) # intersection of discharge times is true positives
        fps = len(pred.difference(gt)) # false positives = dts in pred not in gt
        fns = len(gt.difference(pred)) # false negatives = dts in gt not in pred
        scores['sensitivity'][mu_idx] = tps / (tps + fns) # how real spikes are missed
        scores['precision'][mu_idx] = tps / (tps + fps) # how many fake spikes are assumed
    return scores

# def spike_match_jitter(dt_pred, dts, jitter=4):
#     '''Approximately 2ms of threshold to consider a match'''
#     dt_preds = [dt_pred + jit for jit in range(-jitter, jitter + 1)]
#     time_matches = [dt in dts for dt in dt_preds]
#     return any(time_matches)

def spike_match_jitter(dt_pred, dts, jitter=4):
    '''
    Check if a predicted spike time matches any ground truth spike within a jitter window.
    
    Args:
        dt_pred: Single prediction time point
        dts: Array of ground truth spike times
        jitter: Number of time steps to check in each direction
    '''
    # Vectorized version of the jitter check
    jitter_range = np.arange(-jitter, jitter + 1)
    dt_preds = dt_pred + jitter_range[:, None]  # Broadcasting to create all jittered times
    # Check if any jittered time matches any ground truth time
    matches = np.isin(dt_preds, dts)
    return np.any(matches)

def spike_matching(dts, dts_pred):
    ''' For each motor unit, compute the spiking accuracy, sensitivity and precision.'''
    # matches = [] # indices of the edited spikes that corresponds to the given predicted spike
    # match_scores = [] # score for the found matches
    precisions, sensitivities = torch.zeros(len(dts_pred), len(dts)), torch.zeros(len(dts_pred), len(dts))
    for idx, dt_pred in tqdm(enumerate(dts_pred)):
        for jdx, dt in enumerate(dts):
            tps = np.sum([spike_match_jitter(spike_time, dt) for spike_time in dt_pred])
            fps = len(dt_pred) - tps
            # fps = np.sum([not spike_match_jitter(spike_time, dt) for spike_time in dt_pred]) # false positives = dts in pred not in gt
            fns = np.sum([not spike_match_jitter(spike_time, dt_pred) for spike_time in dt]) # false negatives = dts in gt not in pred
            sensitivities[idx, jdx] = tps / (tps + fns) # how real spikes are missed
            precisions[idx, jdx] = tps / (tps + fps) # how many fake spikes are assumed
    
    # Get best match for each predicted spike with Hungarian algorithm
    f1_scores = 2 * sensitivities * precisions / (sensitivities + precisions + 1e-12)
    print('Linear sum assignment...')
    _, col_ind = linear_sum_assignment((1-f1_scores).numpy())
    matches = [col_ind[idx] for idx in range(len(col_ind))]
    match_scores = [(sensitivities[idx, matches[idx]].item(), precisions[idx, matches[idx]].item()) for idx in range(len(col_ind))]    
    return matches, match_scores

def zero_out_channels(emg_grid, N):
    H, W = emg_grid.shape[2:]

    # Flatten the 2D grid into 1D indices, shuffle, and take first `n`
    indices = torch.randperm(H*W)[:N]
    emg_grid_test = emg_grid.clone().reshape(emg_grid.shape[0], 1, -1)
    emg_grid_test[:, :, indices] = 0
    emg_grid_test = emg_grid_test.reshape(emg_grid.shape[0], 1, H, W)
    return emg_grid_test

if __name__ == '__main__':
    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/2mm'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    R = 16
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
    emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std() + 1e-12) # centering emg
    emg = bandpass_filter(notch_filter(emg, fsamp=fsamp), fsamp=fsamp)
    emg_grid = make_grid(emg, index_matrix)
    Nch = emg_grid.shape[2]*emg_grid.shape[3]

    plt.figure()
    plt.imshow(emg_grid.std(dim=[0,1]))
    # plt.show()
    plt.savefig('grid')

    # Get inverse covariance matrix
    # emg_grid = zero_out_channels(emg_grid, N=250)
    emg_grid[:,:,:, :4] = 0
    emg_grid[:,:,:, 6:] = 0
    # extended_emg_template = np.zeros((R*Nch, emg.shape[1] + R - 1))
    # extended_emg = torch.tensor(extend_emg(extended_emg_template, emg, R))#.to(torch.float32)
    extended_emg = extend_emg_torch(emg_grid.squeeze().reshape(emg_grid.shape[0], -1), R).T
    # inv_cov = get_inv_cov(extended_emg, explained_var=1.0-1e-14)
    inv_cov = get_inv_cov_torch(extended_emg, explained_var=1.0-1e-8)

    # Get separation matrix
    dts = edition['Dischargetimes']
    mu_dts = squeeze_dts(dts)
    mu_dts = filter_dts(mu_dts, start, end)
    # # mu_dts = [mu_dts[idx] for idx in np.random.choice(len(mu_dts), size=20, replace=False)]
    # sep_mat = get_sep_mat(extended_emg, mu_dts) # get separation matrix and test for each dimension it is working
    # sep_mat = get_sep_mat_pseudo_inv(extended_emg, mu_dts, rcond=1e-6) # to avoid whitening
    sep_mat = get_sep_mat_torch(extended_emg, mu_dts)
    # sep_mat = torch.tensor(sep_mat @ inv_cov).to(torch.float32)
    sep_mat = sep_mat @ inv_cov

    # Testing how the number of zeroed out channels affects the performance
    H, W = emg_grid.shape[2:]
    sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=sep_mat, extension_factor=R)

    with torch.no_grad():
        source_est = sda(emg_grid)
    
    pred_dts, sils = get_silohuette(source_est)
    matches, match_scores = spike_matching(mu_dts, pred_dts)
    

    sensitivities = []
    precision = []
    # for N in tqdm([0, 5, 10, 20, 40, 80, 160]):
    #     emg_grid_test = zero_out_channels(emg_grid, N=N)

    #     with torch.no_grad():
    #         source_est = sda(emg_grid_test)
        
    #     pred_dts, sils = get_silohuette(source_est)
    #     matches, match_scores = spike_matching(mu_dts, pred_dts)
    #     print(match_scores)
    #     sensitivities.append(np.mean([match_scores[idx][0] for idx in range(len(match_scores))]))
    #     precision.append(np.mean([match_scores[idx][1] for idx in range(len(match_scores))]))

    # df = pd.DataFrame({'sensitivity': sensitivities, 'precision': precision})
    # df.to_csv('zero_scores.csv', index=False)
