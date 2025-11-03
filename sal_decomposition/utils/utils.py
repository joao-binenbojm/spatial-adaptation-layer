import numpy as np
import scipy
from scipy import signal
from scipy.io import loadmat
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from math import floor
from scipy.ndimage import center_of_mass

from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg, get_silohuette, maxk, bandpass_filter, notch_filter
from sal_decomposition.sda import SpatialDecompositionAdaptation
from loss_functions import KurtosisLoss, NegentropyLoss
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay

def get_theta(input_shape, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, inverse=False):
    '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''

    N, C, H, W = input_shape
    Tx, Ty = torch.tensor(2*Tx/(W-1)), torch.tensor(2*Ty/(H-1)) # Normalize translation values automatically
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
    if inverse:
        theta = torch.linalg.inv(theta)
    return theta

def get_grid(input_shape, theta):
    """ Gets the output sampling grid based on transformation parameters."""
    N, C, H, W = input_shape
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(N,1,1)
    grid = torch.nn.functional.affine_grid(theta, size=(N,C,H,W), align_corners=True)
    return grid

def apply_affine(emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, mode='bilinear'):
    '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''

    theta = get_theta(emg_grid.shape, Tx, Ty, theta, xscale, yscale)
    grid = get_grid(emg_grid.shape, theta)
    xresamp = torch.nn.functional.grid_sample(emg_grid, grid, mode=mode, align_corners=True)
    
    return xresamp

def get_distance(input_shape, theta1, theta2):
    """ Compute Euclidean distance between two affine transformations represented by theta matrices."""
    theta_net = theta1 @ torch.linalg.inv(theta2)

    original_grid = get_grid(input_shape, torch.eye(3))
    original_grid[:,:,:,0] = original_grid[:,:,:,0]*(input_shape[3]-1)/2
    original_grid[:,:,:,1] = original_grid[:,:,:,1]*(input_shape[2]-1)/2

    net_grid = get_grid(input_shape, theta_net)
    net_grid[:,:,:,0] = net_grid[:,:,:,0]*(input_shape[3]-1)/2
    net_grid[:,:,:,1] = net_grid[:,:,:,1]*(input_shape[2]-1)/2

    dist = torch.sqrt(((original_grid - net_grid)**2).sum(dim=-1)).mean().item()
    return dist


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
    d = len(boundaries)
    engine = scipy.stats.qmc.LatinHypercube(d=d)
    init_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float64)-1 # scale from [0,1] to [-1, 1]
    init_params[:,0] = 2*boundaries[0]*init_params[:,0]/(W-1)
    if d > 1:
        init_params[:,1] = 2*boundaries[1]*init_params[:,1]/(H-1)
    if d > 2:
        init_params[:,2] = boundaries[2]*init_params[:, 2]/np.pi
    if d > 3:
        init_params[:, 3] = (boundaries[3] - 1/boundaries[3])*(init_params[:, 3] + 1 )/2 + 1/boundaries[3]
    if d > 4:
        init_params[:, 4] = (boundaries[4] - 1/boundaries[4])*(init_params[:, 4] + 1 )/2 + 1/boundaries[4]

    init_params = init_params.to(device)
    with torch.no_grad():
        
        for npoint in tqdm(range(npoints)):
            # Set initial conditions
            sda.sal.xshift[0].copy_(init_params[npoint, 0])
            if d > 1:
                sda.sal.yshift[0].copy_(init_params[npoint, 1])
            if d > 2:
                sda.sal.rot_theta[0].copy_(init_params[npoint, 2])
            if d > 3:
                sda.sal.xscale[0].copy_(init_params[npoint, 3])
            if d > 4:
                sda.sal.yscale[0].copy_(init_params[npoint, 4])

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
            # sda.sal.xshift[0].data, sda.sal.yshift[0].data, sda.sal.rot_theta[0].data = init_params[losses.argmax(), :3] # get best initialization
            # sda.sal.xscale.data, sda.sal.yscale.data = init_params[losses.argmax(), 3:]
            sda.sal.xshift[0].copy_(init_params[losses.argmax(), 0]) # get best initialization
            if d > 1:
                sda.sal.yshift[0].copy_(init_params[losses.argmax(), 1])
            if d > 2:
                sda.sal.rot_theta[0].copy_(init_params[losses.argmax(), 2])
            if d > 3:
                sda.sal.xscale[0].copy_(init_params[losses.argmax(), 3])
            if d > 4:
                sda.sal.yscale[0].copy_(init_params[losses.argmax(), 4])
            
            print(f'TOP 5 LOSS VALUES SAMPLED: {torch.topk(losses, k=torch.min(torch.tensor([npoints, 5])))}')

    # Make SAL parameters learnable
    if frozen_sep_mat:
        for param in sda.sal.parameters():
            param.requires_grad = True        
    else:
        for param in sda.parameters():
            param.requires_grad = True

    # Make channel scalings learnable
    # sda.channel_scales.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sda.parameters()),
                                                lr=lr)        

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
        print(f'PARAMS:\n xshift: {(W-1)*sda.sal.xshift[0].item()/2}, yshift: {(H-1)*sda.sal.yshift[0].item()/2}, theta: {sda.sal.rot_theta[0].item()} ')
        # print(f'xscale: {sda.sal.xscale.item()}, yscale: {sda.sal.yscale.item()}')
        # Collect outputs and loss
        losses.append(epoch_loss)

    # Get final outputs, i.e. optimal souces
    with torch.no_grad():
        batch_outputs = []
        N = emg_grid_transform.shape[0]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch = emg_grid_transform[start_idx:end_idx].to(device)
            batch_out = sda(batch)
            batch_outputs.append(batch_out.cpu())  # Move to CPU to save GPU RAM
        sources = torch.cat(batch_outputs, dim=0)

    return sources, losses

# def loss_sampling(emg_grid_transform, sda, base_loss=1.0, T=(0.0, 0.0), bounds=(0.0, 0.0), batch_size=2048, num_points=20, loss='kurtosis', device='cpu'):
#     ''' Method used to sample the loss landscape.'''
#     N, C, H, W = emg_grid_transform.shape
#     if loss == 'kurtosis':
#         ica_loss = KurtosisLoss()
#     else:
#         ica_loss = NegentropyLoss()

#     # Getting torch meshgrid
#     if not T:
#         Tx, Ty = W//2, H//2
#     else:
#         Tx, Ty = T
#     xbounds, ybounds = bounds
#     x = torch.linspace(-torch.tensor(xbounds), torch.tensor(xbounds), num_points)
#     y = torch.linspace(-torch.tensor(ybounds), torch.tensor(ybounds), num_points)
#     xx, yy = torch.meshgrid(x, y, indexing='xy')
    
#     # Reshape grid points into a batch
#     xflat = xx.reshape(-1).to(device)
#     yflat = yy.reshape(-1).to(device)
    
#     # Sample parameters
#     sda.train()
#     with torch.no_grad():
#         # Process grid points in batches
#         grid_batch_size = 100  # Number of grid points to process at once
#         total_points = len(xflat)
#         loss_flat = torch.zeros(total_points, device=device)
        
#         for grid_idx in tqdm(range(0, total_points, grid_batch_size)):
#             grid_end_idx = min(grid_idx + grid_batch_size, total_points)
            
#             # Set shifts for this batch of grid points
#             xshifts_batch = 2 * xflat[grid_idx:grid_end_idx] / (W-1)
#             yshifts_batch = 2 * yflat[grid_idx:grid_end_idx] / (H-1)
            
#             batch_losses = []
#             # Process EMG data in batches for each grid point
#             for start_idx in range(0, N, batch_size):
#                 end_idx = min(start_idx + batch_size, N)
#                 emg_batch = emg_grid_transform[start_idx:end_idx].to(device)
                
#                 # Compute losses for all grid points in current batch
#                 batch_outputs = []
#                 for xshift, yshift in zip(xshifts_batch, yshifts_batch):
#                     sda.sal.xshift[0].copy_(xshift)
#                     sda.sal.yshift[0].copy_(yshift)
#                     batch_output = sda(emg_batch)
#                     batch_outputs.append(ica_loss(batch_output))
                
#                 batch_losses.append(batch_outputs)
            
#             # Average losses across EMG batches
#             avg_losses = torch.tensor(batch_losses).mean(dim=0)
#             loss_flat[grid_idx:grid_end_idx] = avg_losses

#     # Reshape losses back to grid
#     loss_arr = loss_flat.reshape(num_points, num_points)

#     plt.figure()
#     ax = sns.heatmap(np.array(loss_arr.cpu())/base_loss)
#     ax.set(xlabel='Circumferential Shifts (mm)', ylabel='Longitudinal Shifts (mm)')
#     if T:
#         ax.text(np.where(np.array(x.cpu())>=Tx)[0][0] + 0.5, 
#                 np.where(y.cpu()>=Ty)[0][0]+0.5, 'X', 
#             color='green', ha='center', va='center', fontsize=16)
    
#     plt.savefig('loss_landscape.jpg')
#     print()

#     return loss_arr

def loss_sampling(emg_grid_transform, sda, base_loss=1.0, T=(0.0, 0.0), bounds=(0.0, 0.0), 
                  batch_size=2048, num_points=20, loss='kurtosis', device='cpu'):
    ''' Method used to sample the loss landscape.'''
    N, C, H, W = emg_grid_transform.shape
    
    if loss == 'kurtosis':
        ica_loss = KurtosisLoss()
    else:
        ica_loss = NegentropyLoss()

    # Getting torch meshgrid
    if not T:
        Tx, Ty = W//2, H//2
    else:
        Tx, Ty = T
    xbounds, ybounds = bounds
    x = torch.linspace(-torch.tensor(xbounds), torch.tensor(xbounds), num_points)
    y = torch.linspace(-torch.tensor(ybounds), torch.tensor(ybounds), num_points)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Reshape grid points into a batch
    xflat = xx.reshape(-1).to(device)
    yflat = yy.reshape(-1).to(device)
    
    # Sample parameters
    sda.eval()  # Use eval mode for consistent results
    with torch.no_grad():
        total_points = len(xflat)
        loss_flat = torch.zeros(total_points, device=device)
        
        # For each grid point
        for grid_idx in tqdm(range(total_points)):
            # Set shift for this grid point
            xshift = 2 * xflat[grid_idx] / (W-1)
            yshift = 2 * yflat[grid_idx] / (H-1)
            
            sda.sal.xshift[0].copy_(xshift)
            sda.sal.yshift[0].copy_(yshift)
            
            # Accumulate loss across all EMG data batches
            total_loss = 0.0
            num_samples = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                emg_batch = emg_grid_transform[start_idx:end_idx].to(device)
                
                # Apply transformation and compute loss
                batch_output = sda(emg_batch)
                batch_loss = ica_loss(batch_output)
                
                # Check if loss is per-sample or already reduced
                if batch_loss.dim() == 0:  # Scalar loss
                    # Loss is already averaged over batch
                    total_loss += batch_loss.item() * (end_idx - start_idx)
                    num_samples += (end_idx - start_idx)
                else:  # Per-sample losses
                    total_loss += batch_loss.sum().item()
                    num_samples += batch_loss.numel()
                
                # Free GPU memory
                del emg_batch, batch_output, batch_loss
                torch.cuda.empty_cache()
            
            # Average loss across all samples for this grid point
            loss_flat[grid_idx] = total_loss / num_samples

    # Reshape losses back to grid
    loss_arr = loss_flat.reshape(num_points, num_points)

    plt.figure()
    ax = sns.heatmap(np.array(loss_arr.cpu())/base_loss)
    ax.set(xlabel='Circumferential Shifts (mm)', ylabel='Longitudinal Shifts (mm)')
    if T:
        ax.text(np.where(np.array(x.cpu())>=Tx)[0][0] + 0.5, 
                np.where(y.cpu()>=Ty)[0][0]+0.5, 'X', 
            color='green', ha='center', va='center', fontsize=16)
    
    plt.savefig('loss_landscape.jpg')
    print()

    return loss_arr


def get_transformed_grid(grid_shape, Tx=0, Ty=0, theta=0, xscale=1, yscale=1):
    '''Computes the transformed grid coordinates for euclidina distance comparison.'''

    N, C, H, W = grid_shape
    Tx, Ty = torch.tensor(2*Tx/(W-1)), torch.tensor(2*Ty/(H-1)) # Normalize translation values automatically
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

    # Obtain transformed grid in pixel units
    grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=True)
    grid[:,:,:,0] = (W-1)*(1 + grid[:,:,:,0])/2
    grid[:,:,:,1] = (H-1)*(1 + grid[:,:,:,1])/2
    return grid

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
    cov_mat = cov_mat + reg*torch.eye(cov_mat.shape[0], device=signal.device).to(torch.float64)
    evalues, evectors  = torch.linalg.eigh(cov_mat)
    sorted_idxs = torch.argsort(evalues, descending=True)
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    inv_cov = evectors @ torch.diag(1 / (evalues)) @ evectors.T
    return inv_cov

def whitening_torch(signal, explained_var=0.99):
    ''' Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization all in PyTorch (compute in torch.float64 to avoid precision issues). '''
    cov_mat = torch.cov(signal.to(torch.float64))
    evalues, evectors  = torch.linalg.eigh(cov_mat)
    sorted_idxs = torch.argsort(evalues, descending=True)
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    cum_explained_var = evalues.cumsum(dim=0) / evalues.sum()
    evalues, evectors = evalues[cum_explained_var <= explained_var], evectors[:, cum_explained_var <= explained_var]
    whit_mat = evectors @ torch.diag(1 / torch.sqrt((evalues))) @ evectors.T
    return whit_mat @ signal, whit_mat

def whitening_tikhonov(signal, reg=1e-6):
    ''' Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization all in PyTorch (compute in torch.float64 to avoid precision issues). '''
    cov_mat = torch.cov(signal.to(torch.float64))
    cov_mat = cov_mat + reg*torch.eye(cov_mat.shape[0]).to(torch.float64)
    evalues, evectors  = torch.linalg.eigh(cov_mat)
    sorted_idxs = torch.argsort(evalues, descending=True)
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    whit_mat = evectors @ torch.diag(1 / torch.sqrt((evalues))) @ evectors.T
    return whit_mat @ signal

def open_mat_output(DIR, name):
    ''' Open data from Emanuele's files so that it can be further processed'''

    try:
        arr = loadmat(os.path.join(DIR, name), struct_as_record=False, mat_dtype=True)
            # Load acquisition signal
        signal, edition = {}, {}
        for attr in dir(arr['signal'][0,0]):
            if attr[0] != '_': # only data relevant attributes
                signal[attr] = getattr(arr['signal'][0,0], attr)

        for attr in dir(arr['edition'][0,0]):
            if attr[0] != '_': # only data relevant attributes
                edition[attr] = getattr(arr['edition'][0,0], attr)
    # In case the matlab file is 
    except:
        signal, edition = {}, {}
        with h5py.File(os.path.join(DIR, name), 'r') as f:
            signal['data'] = f['signal']['data'][:, :256].T
            signal['target'] = f['signal']['target'][:]
            signal['path'] = f['signal']['path'][:]
            dts = []
            for idx in range(4):
                ref = f['edition']['Distimeclean'][idx, 0]
                for pt_idx in range(f[ref].shape[0]):
                    if len(f[ref].shape) == 2:
                        subref = f[ref][pt_idx, 0]
                        dts.append(f[subref][:].squeeze())
                    else:
                        subref = f[ref][pt_idx]
            edition['Dischargetimes'] = dts    

    return signal, edition

def get_target_boundaries(target, threshold=0.9):
    '''Takes in signal and edition dictionaries and returns the data and edition dictionaries with the target timestamps only.'''
    target_max = np.max(target)
    # Find rising and falling edges of target signal
    rising_edge = np.where(target >= threshold*target_max)[0][0] + 1000
    falling_edge = np.where(target >= threshold*target_max)[0][-1] - 1000
 
    return rising_edge, falling_edge

def get_mean_firing_rate(dts, fs):
    """ Given the discharge times within the plateuau of the task, compute the firing rate in Hz."""
    rates = fs / np.array([np.diff(dt).mean() for dt in dts])
    return rates

def get_median_firing_rate(dts, fs):
    """ Given the discharge times within the plateuau of the task, compute the firing rate in Hz."""
    rates = fs / np.array([np.median(np.diff(dt)) for dt in dts])
    return rates

def get_cov_isi(dts):
    """ Given the discharge times within the plateau, computes the variability of ISIs per unit."""
    isi_mean = np.array([np.mean(np.diff(dt)) for dt in dts])
    isi_std = np.array([np.std(np.diff(dt)) for dt in dts])
    cov = isi_std / (isi_mean + 1e-12)
    return cov

def get_trimmed_cov_isi(dts, trim_percent=0.01):
    """
    Computes the CoV of ISIs after trimming a percentage of
    outliers from both ends of the ISI distribution.

    This is robust to single missed/false spikes that create
    artificially large or small ISIs.

    Args:
        dts: A list of 1D numpy arrays, where each array
             contains the discharge times for one motor unit.
        trim_percent: The percentage of ISIs to trim from *each* end
                      of the sorted distribution.
                      - 0.05 (default) trims the 5% smallest and 5%
                        largest ISIs (10% total).
                      - 0.1 trims 10% from each end (20% total).

    Returns:
        A 1D numpy array of the trimmed CoV for each unit.
    """
    trimmed_covs = []
    
    for dt in dts:
        # Need at least 3 spikes to get 2 ISIs
        if len(dt) < 3:
            trimmed_covs.append(np.nan)
            continue
            
        isis = np.diff(dt)
        n_isis = len(isis)
        
        # Need at least 2 ISIs to calculate std
        if n_isis < 2:
            trimmed_covs.append(np.nan)
            continue

        # Sort ISIs to find and remove extremes
        isis_sorted = np.sort(isis)
        
        # Calculate how many ISIs to cut from each end
        n_to_trim = int(n_isis * trim_percent)
        
        # Slice the array to get the "clean" data
        # Note: If n_to_trim is 0, this just returns the full array
        if n_to_trim > 0:
            trimmed_isis = isis_sorted[n_to_trim:-n_to_trim]
        else:
            trimmed_isis = isis_sorted # Use all ISIs if trimming is not possible

        # If we trimmed everything (or too much), we can't get a CoV
        if len(trimmed_isis) < 2:
            trimmed_covs.append(np.nan)
            continue

        # Calculate mean and std on the *trimmed* data
        trimmed_mean = np.mean(trimmed_isis)
        trimmed_std = np.std(trimmed_isis)
        
        # Avoid division by zero
        if trimmed_mean < 1e-12:
            trimmed_covs.append(np.nan)
            continue
            
        cov = trimmed_std / (trimmed_mean + 1e-12)
        trimmed_covs.append(cov)
        
    return np.array(trimmed_covs)

def get_recruitment_threshold(dts, target, N=10):
    """Estimate recruitment threshold (%MVC) based on average of the initial 10 discharge times."""
    rts = np.zeros(len(dts))
    start = np.where(target > 1)[0][0]
    for dt_idx, dt in enumerate(dts):
        dummy_dt = dt.copy()
        dummy_dt = dummy_dt[dummy_dt >= start]
        rts[dt_idx] = np.mean(target[dummy_dt[:N].astype(int)])
    return rts

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
    if isinstance(dts, list):
        return dts

    mu_dts = []
    for idx in range(dts.shape[0]):
        for jdx in range(dts.shape[1]):
            # Get non-zero discharge times for this MU
            valid_dts = dts[idx, jdx]
            if valid_dts.shape[1] > 0:  # Only add if there are valid discharge times
                mu_dts.append(valid_dts.squeeze())
    return mu_dts

def make_grid(emg, index_matrix, ied=2):
    '''Function that converts EMG grid into the dimensions of a batch of images, expected by the affine transforms and decomposition module. (Input shape H, W, T)'''
    emg_grid = torch.tensor(emg[index_matrix, :]).to(torch.float64)

    # # Add cornel pixel as average of 3 neighbours
    # if ied == 2:
    #     emg_grid[0,0,:] = (emg_grid[0,1,:] + emg_grid[1,0,:] + emg_grid[1,1,:])/3
    #     emg_grid[0,-1,:] = (emg_grid[0,-2,:] + emg_grid[1,-1,:] + emg_grid[1,-2,:])/3
    #     emg_grid[-1,0,:] = (emg_grid[-2,0,:] + emg_grid[-1,1,:] + emg_grid[-2,1,:])/3
    #     emg_grid[-1,-1,:] = (emg_grid[-2,-1,:] + emg_grid[-1,-2,:] + emg_grid[-2,-2,:])/3

    emg_grid = emg_grid.permute(2, 0, 1).unsqueeze(1)
    return emg_grid

def get_sta_templates(extended_emg, dts):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    N = len(dts) # number of MUs
    sep_mat = torch.zeros((N, extended_emg.shape[0])).to(torch.float64)
    for idx in range(N):
        sep_mat[idx, :] = (extended_emg[:, dts[idx].astype(int)]).mean(dim=1)
        sep_mat[idx, :] = sep_mat[idx, :] / (torch.norm(sep_mat[idx, :]) + 1e-12)
    return sep_mat

def get_sep_mat_pseudo_inv(extended_emg, dts, rcond=1e-3):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    N = len(dts) # number of MUs
    y_inv = torch.linalg.pinv(extended_emg, rcond=rcond)
    spike_trains = torch.zeros((N, extended_emg.shape[1])).to(torch.float64)
    for idx in range(N):
        spike_trains[idx, dts[idx]] = 1.0
    sep_mat = spike_trains @ y_inv
    sep_mat = sep_mat / (torch.norm(sep_mat, dim=1, keepdim=True)**2 + 1e-10) # ensure each sep_mat row has norm 1
    return sep_mat

def get_spectral_flatness_ar2(data, fs=2048, n_fft=512):
    """
    Compute spectral flatness (Wiener entropy) from AR(2) model for each channel.

    Parameters:
        data: np.ndarray of shape (T, Ch) - input signal window
        fs: Sampling frequency in Hz
        n_fft: Number of frequency points to evaluate PSD

    Returns:
        flatness: np.ndarray of shape (1, Ch)
    """
    T, H, W = data.shape
    flatness = np.zeros((H, W))
    freqs = np.linspace(0, fs / 2, n_fft)

    from statsmodels.regression.linear_model import yule_walker

    for h in range(H):
        for w in range(W):
            x = data[:, h, w]

            try:
                ar_coeffs, sigma2 = yule_walker(x, order=2, method='mle')
            except Exception:
                flatness[h, w] = 1.0  # fallback: flat
                continue

            # Evaluate PSD over frequency grid using parametric AR model
            a = np.concatenate([[1], -ar_coeffs])  # AR polynomial
            omega = 2 * np.pi * freqs / fs
            exp_terms = np.exp(-1j * np.outer(omega, np.arange(len(a))))
            H = 1 / (exp_terms @ a)
            psd = sigma2 * np.abs(H) ** 2

            # Spectral flatness: geometric mean / arithmetic mean (Wiener entropy)
            psd = np.maximum(psd, 1e-12)  # to avoid log(0)
            geo_mean = np.exp(np.mean(np.log(psd)))
            arith_mean = np.mean(psd)
            flatness[h, w] = geo_mean / arith_mean

    return flatness

def get_sta_muaps(emg_grid, discharge_times, L, spacing=15, plot=True):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    T, _, H, W = emg_grid.shape
    if not isinstance(discharge_times, torch.Tensor):
        discharge_times = torch.tensor(discharge_times, dtype=torch.long, device=emg_grid.device)
    else:
        # If it's already a tensor, make sure it's the right type
        discharge_times = discharge_times.to(dtype=torch.long)

    # Keep only valid spikes (so window fits)
    valid_times = discharge_times[
        (discharge_times >= L) & (discharge_times < T - L + 1)
    ]

    if len(valid_times) == 0:
        return torch.zeros(L, H, W, device=emg_grid.device)

    # Collect snippets: (n_spikes, L, 1, H, W)
    snippets = torch.stack([
        emg_grid[i-L:i+L+1] for i in valid_times if emg_grid[i-L:i+L+1].shape[0] == 2*L + 1
    ], dim=0)

    # Average across spikes → (L, H, W)
    sta = snippets.mean(dim=0).squeeze(1)

    # Plot in the same grid in a tiled fashio
    time = np.arange(2*L + 1)
    if plot:

        # If plotting, the normalise so we can better visualize muaps
        # sta = (sta - sta.mean()) / (sta.std() + 1e-9)
        sta = (sta - sta.min()) / (sta.max() - sta.min() + 1e-9)
        plt.figure(figsize=(W/2, H/2))
        
        for h in range(H):
            for w in range(W):
                y = sta[:, h, w]
                # shift by electrode position
                y_offset = (H-1-h) * spacing
                x_offset = w * (2*L + 1) * spacing / W  # scale horizontally
                plt.plot(time + x_offset, 15.0*y + y_offset, color="k", lw=0.6)
                # plt.vlines((time + x_offset), )

        plt.axis("off")
        plt.title("MUAP waveforms (STA)")
        plt.savefig('muaps.jpg')
        plt.close()
    return sta

def get_p2p_muaps(emg_grid, discharge_times, L, plot=True):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    sta = get_sta_muaps(emg_grid, discharge_times, L, plot=False)
    p2p = np.max(sta.numpy(), axis=0) - np.min(sta.numpy(), axis=0)
    if plot:
        plt.figure()
        sns.heatmap(p2p)
        plt.savefig('peak2peak')
        plt.close()

    return p2p

def get_p2p_com(emg_grid, discharge_times, L):
    """ Takes in emg grid and computes the P2P CoM, so we get a rough idea for a given MUAPs location."""
    p2p = get_p2p_muaps(emg_grid, discharge_times, L, plot=False)
    cy_scipy, cx_scipy = center_of_mass(np.array(p2p))
    return cy_scipy, cx_scipy

def get_sta_correlations(emg_grid, dts1, dts2, L, get_p=False):
    """Aligns temporally, and assumes computes correlation between the STA of two separate spike trains."""
    H, W = emg_grid.shape[2:]

    dts1, dts2 = dts1.copy(), dts2.copy() # prevents aliasing when we edit the discharge times

    # Finds highest energy point of best channel to align muaps
    shifts = []
    for dts in [dts1, dts2]:
        sta = get_sta_muaps(emg_grid.clone(), dts, L, plot=False)
        p2p = np.max(sta.numpy(), axis=0) - np.min(sta.numpy(), axis=0)
        flat_idx = np.argmax(p2p).astype(int)
        r, c = flat_idx // W, flat_idx % W
        best_channel_abs = torch.abs(sta[:, r, c])
        shifts.append(torch.argmax(best_channel_abs).item())

    # Aligning discharge times
    dts1 = np.array(dts1) - shifts[0]
    dts2 = np.array(dts2) - shifts[1]

    sta1, sta2 = get_sta_muaps(emg_grid.clone(), dts1, L, plot=False), get_sta_muaps(emg_grid.clone(), dts2, L, plot=False)
    sta1, sta2 = sta1.numpy().flatten(), sta2.numpy().flatten()

    corr, p = scipy.stats.pearsonr(sta1, sta2) # computes pearson correlation and pearson correlation coefficient

    if get_p:
        return corr, p
    else:
        return corr

def get_stv(emg_grid, discharge_times, L):
    '''Takes in extended EMG and dischage times from different MUs and returns separation matrix all in PyTorch.'''
    T, _, H, W = emg_grid.shape
    if isinstance(discharge_times, list):
        discharge_times = torch.tensor(discharge_times, device=emg_grid.device)

    # Keep only valid spikes (so window fits)
    valid_times = discharge_times[
        (discharge_times >= L) & (discharge_times < T - L + 1)
    ]

    if len(valid_times) == 0:
        return torch.zeros(L, H, W, device=emg_grid.device)

    # Collect snippets: (n_spikes, L, 1, H, W)
    snippets = torch.stack([
        emg_grid[i-L:i+L+1] for i in valid_times if emg_grid[i-L:i+L+1].shape[0] == 2*L + 1
    ], dim=0)

    # Average across spikes → (L, H, W)
    stv = snippets.std(dim=0).squeeze(1)
    return stv.mean().item()


def plot_grid_psd(emg_grid, fs=2048, n_fft=512, spacing=1.2, plot=True):
    """
    Plots the PSD of each channel in the EMG grid in a tiled grid format.

    Args:
        emg_grid (torch.Tensor): Shape (T, 1, H, W)
        fs (int): Sampling frequency in Hz
        n_fft (int): Number of FFT points for PSD
        spacing (float): Spacing for grid plot
        plot (bool): Whether to plot and save the figure

    Returns:
        psd_grid (np.ndarray): PSD values for each channel, shape (n_freqs, H, W)
        freqs (np.ndarray): Frequency axis
    """
    T, _, H, W = emg_grid.shape
    emg_np = emg_grid.squeeze(1).cpu().numpy()  # Shape: (T, H, W)
    psd_grid = np.zeros((n_fft//2+1, H, W))
    freqs = None

    for h in range(H):
        for w in range(W):
            f, Pxx = scipy.signal.welch(emg_np[:, h, w], fs=fs, nperseg=256, nfft=n_fft)
            psd_grid[:, h, w] = Pxx
            if freqs is None:
                freqs = f

    if plot:
        plt.figure(figsize=(W/2, H/2))
        for h in range(H):
            for w in range(W):
                y = 10 * np.log10(psd_grid[:, h, w] + 1e-12)
                y_offset = (H-1-h) * spacing
                x_offset = w * len(freqs) * spacing / W
                plt.plot(freqs + x_offset, y + y_offset, color="b", lw=0.6)
        plt.axis("off")
        plt.title("PSD of EMG Channels (Welch)")
        plt.savefig('grid_psd.jpg')
    return psd_grid, freqs

def plot_channel_psd(emg_grid, h, w, fs=2048, n_fft=512, plot=True):
    """
    Plots the PSD of a single channel (h, w) from the EMG grid.

    Args:
        emg_grid (torch.Tensor): Shape (T, 1, H, W)
        h (int): Row index of the channel
        w (int): Column index of the channel
        fs (int): Sampling frequency in Hz
        n_fft (int): Number of FFT points for PSD
        plot (bool): Whether to plot and show the figure

    Returns:
        freqs (np.ndarray): Frequency axis
        psd (np.ndarray): PSD values for the selected channel
    """
    T, _, H, W = emg_grid.shape
    emg_np = emg_grid.squeeze(1).cpu().numpy()  # Shape: (T, H, W)
    f, Pxx = scipy.signal.welch(emg_np[:, h, w], fs=fs, nperseg=256, nfft=n_fft)

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(f[:len(f)//3], Pxx[:len(f)//3], color="b", lw=1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"PSD of Channel ({h}, {w})")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('psd')

    return f, Pxx

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
    return extended_emg[:-(R-1),:]

# def get_silohuette(sources_pred, distance=4):
#     '''Get silhouette values given source predictions.'''
    
#     # Step 4b:
#     sils = np.zeros(sources_pred.shape[1])
#     pred_dts = []
#     for mu_idx in range(sources_pred.shape[1]):
#         source_pred = sources_pred[:, mu_idx] # get a single source prediction
#         source_pred = np.multiply(source_pred, source_pred) # get squared sources
#         peaks, _ = scipy.signal.find_peaks(source_pred.squeeze(), distance=distance) # default about 2ms 
#         source_pred /=  np.mean(maxk(source_pred[peaks], 10))
#         if len(peaks) > 1:
#             kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
#             # indices of the spike and noise clusters (the spike cluster should have a larger value)
#             spikes_ind = np.argmax(kmeans.cluster_centers_)
#             noise_ind = np.argmin(kmeans.cluster_centers_)
#             # get the points that correspond to each of these clusters
#             spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
#             noise = peaks[np.where(kmeans.labels_ == noise_ind)]
#             # calculate the centroids
#             spikes_centroid = kmeans.cluster_centers_[spikes_ind]
#             noise_centroid = kmeans.cluster_centers_[noise_ind]
#             # difference between the within-cluster sums of point-to-centroid distances 
#             intra_sums = (((source_pred[spikes]- spikes_centroid)**2).sum()) 
#             # difference between the between-cluster sums of point-to-centroid distances
#             inter_sums = (((source_pred[spikes] - noise_centroid)**2).sum())
#             sil = (inter_sums - intra_sums) / max(intra_sums, inter_sums)  
#         else:
#             sil = 0
#             spikes = np.array([])
#         sils[mu_idx] = sil
#         pred_dts.append(spikes)
#     return pred_dts, sils

import torch
import torch.nn.functional as F

def get_sta_templates_peeloff(emg_grid, discharge_times_list, R, L=20):
    """
    Update separation vectors using sequential peeling (no new discharge detection).
    
    Args:
        emg_grid: (T, 1, H, W) original grid format EMG
        discharge_times_list: list of N discharge time arrays (one per MU)
        L: template length for MUAP
        R: extension factor for extended observation
        
    Returns:
        separation_vectors: (N, nchans*R) updated separation vectors
        muap_templates: list of N MUAP templates (L, H, W)
    """
    T, _, H, W = emg_grid.shape
    nchans = H * W
    N = len(discharge_times_list)
    device = emg_grid.device
    
    # Flatten and extend the original EMG
    emg_flat = emg_grid.squeeze(1).reshape(T, nchans)  # (T, nchans)
    extended_emg = extend_emg_torch(emg_flat, R)  # (T, nchans*R)
    extended_emg_residual = extended_emg.clone()
    
    # Initialize outputs
    separation_vectors = torch.zeros((N, nchans * R), dtype=torch.float64, device=device)
    muap_templates = []
        
    # Sequential peeling (no ordering, just go through MUs in order)
    for mu_idx in tqdm(range(N)):
        dts = torch.tensor(discharge_times_list[mu_idx], device=device)
                
        # Keep only valid times for this MU
        valid_times = dts[(dts >= L) & (dts < T - L)].long()

        if len(valid_times) == 0:
            muap_templates.append(torch.zeros(L, H, W, device=device))
            continue
        
        # 1. Compute separation vector: STA at spike times in extended space (no windowing)
        sep_vec = extended_emg_residual[valid_times, :].mean(dim=0)  # (nchans*R,)
        sep_vec = sep_vec / (torch.norm(sep_vec) + 1e-12)
        separation_vectors[mu_idx] = sep_vec
        
        # 2. Compute MUAP template on current residual for peeling
        emg_residual_flat = extended_emg_residual[:, :nchans]  # Take delay-0 channels
        emg_residual_grid = emg_residual_flat.reshape(T, 1, H, W)
        
        sta_muaps = get_sta_muaps(emg_residual_grid, valid_times, L, plot=False)
        muap_templates.append(sta_muaps)
        
        # 3. Create spike train
        spike_train = torch.zeros(T, device=device, dtype=torch.float64)
        spike_train[valid_times.long()] = 1.0
        spike_train = spike_train.view(1, 1, -1)  # (batch=1, in_ch=1, T)
        
        # 4. Convolve spike train with MUAP template (center aligned)
        sta_flat = sta_muaps.reshape(2*L + 1, nchans)  # (L, nchans)
        sta_flat = torch.flip(sta_flat, dims=[0]) # flip for it to make sense with cross-correlation
        kernel = sta_flat.T.unsqueeze(1)         # (nchans, 1, L)        
        kernel_full = torch.zeros(2*L + 1 + L, nchans, device=sta_flat.device, dtype=sta_flat.dtype)
        # Put the STA template starting at index 'half'
        kernel_full[L:L + 2*L + 1, :] = sta_flat
        kernel_full = kernel_full.T.unsqueeze(1)  # shape (nchans, 1, kernel_size)

        # Conv without padding (so kernel is placed starting at spike index)
        muap_train = F.conv1d(spike_train, kernel, padding='same').squeeze().T  # (1, nchans, T-L+1)
        
        # 5. Extend the MUAP train
        extended_muap_train = extend_emg_torch(muap_train, R)
        
        # 6. Peel: subtract from residual for next MU
        extended_emg_residual = extended_emg_residual - extended_muap_train
    
    return separation_vectors


def get_silohuette(sources_pred, distance=4):
    '''Get silhouette values given source predictions.'''
    
    # Convert torch tensor to numpy if needed
    if isinstance(sources_pred, torch.Tensor):
        sources_pred = sources_pred.cpu().numpy()
    
    sils = np.zeros(sources_pred.shape[1])
    pred_dts = []
    
    for mu_idx in range(sources_pred.shape[1]):
        source_pred = sources_pred[:, mu_idx]
        source_pred_sq = np.multiply(source_pred, abs(source_pred))          
        # Find peaks with only distance constraint
        peaks, _ = scipy.signal.find_peaks(source_pred_sq.squeeze(), distance=distance)
        
        # Normalize
        peak_values = source_pred_sq[peaks]
        
        if len(peak_values) > 0:
            k = min(10, len(peak_values))
            top_k_mean = np.mean(np.partition(peak_values, -k)[-k:])
            if top_k_mean > 0:
                peak_values = peak_values / top_k_mean
        
        # K-means clustering if we have at least 2 peaks
        if len(peaks) >= 2:
            kmeans = KMeans(
                n_clusters=2, 
                init='k-means++', 
                n_init=10
            ).fit(peak_values.reshape(-1, 1))
            
            # Identify spike cluster (higher centroid)
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            noise_ind = np.argmin(kmeans.cluster_centers_)
            
            spikes = peaks[kmeans.labels_ == spikes_ind]
            
            # Calculate silhouette
            spike_vals = peak_values[kmeans.labels_ == spikes_ind]
            noise_vals = peak_values[kmeans.labels_ == noise_ind]
            
            if len(spike_vals) > 0 and len(noise_vals) > 0:
                spikes_centroid = kmeans.cluster_centers_[spikes_ind]
                noise_centroid = kmeans.cluster_centers_[noise_ind]
                
                intra_dist = np.sum(np.square(spike_vals - spikes_centroid))
                inter_dist = np.sum(np.square(spike_vals - noise_centroid))
                
                if max(intra_dist, inter_dist) > 0:
                    sil = (inter_dist - intra_dist) / max(intra_dist, inter_dist)
                else:
                    sil = 0
            else:
                sil = 0
        else:
            # Less than 2 peaks - keep all peaks
            spikes = peaks
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


# def spike_matching(dts, dts_pred, fs, old_matches=None, jitter=0.002):
#     ''' For each motor unit, compute the spiking accuracy, sensitivity and precision.'''
#     precisions = torch.zeros(len(dts_pred), len(dts))
#     sensitivities = torch.zeros(len(dts_pred), len(dts))
#     rate_of_agreement = torch.zeros(len(dts_pred), len(dts)) ## USES THIS TO DETERMINE MATCHES
    
#     # Convert lists to tensors/arrays if they aren't already
#     dts = [torch.tensor(dt) for dt in dts]
#     dts_pred = [torch.tensor(dt_pred) for dt_pred in dts_pred]
    
#     for idx, dt_pred in enumerate(dts_pred):
#         # Vectorize the outer loop by creating a matrix of differences
#         pred_times = dt_pred.reshape(-1, 1)  # Shape: (n_pred, 1)
        
#         for jdx, dt in enumerate(dts):
#             # Broadcast subtraction
#             time_diffs = torch.abs(pred_times.reshape(-1, 1) - dt.reshape(1, -1))  # Shape: (n_pred, n_true)
            
#             # Compute matches using broadcasting
#             matches = (time_diffs <= int(jitter*fs)).any(dim=1)  # Assuming spike_match_jitter threshold is 1
#             tps = matches.sum()
#             fps = len(dt_pred) - tps
            
#             # Compute false negatives using vectorized operations
#             gt_matches = (time_diffs <= int(jitter*fs)).any(dim=0)
#             fns = (~gt_matches).sum()
            
#             if (tps + fns) == 0:
#                 sensitivities[idx, jdx] = 0.0
#             else:
#                 sensitivities[idx, jdx] = tps / (tps + fns)
#             if (tps + fps) == 0:
#                 precisions[idx, jdx] = 0.0
#             else:
#                 precisions[idx, jdx] = tps / (tps + fps)
#             if (tps + fns + fps) == 0:
#                 rate_of_agreement[idx, jdx] = 0.0
#             else:
#                 rate_of_agreement[idx, jdx] = tps / (tps + fns + fps)
    
#     # Get best match for each predicted spike with Hungarian algorithm
#     f1_scores = 2 * sensitivities * precisions / (sensitivities + precisions + 1e-12)
    
#     if old_matches is None:
#         print('Linear sum assignment...')
#         # row_ind, col_ind = linear_sum_assignment((1-f1_scores).numpy())
#         row_ind, col_ind = linear_sum_assignment((1-rate_of_agreement).numpy())
#     else:
#         row_ind, col_ind = np.array(list(old_matches.keys())), np.array(list(old_matches.values()))
    
#     # Vectorize final computations
#     matches = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}

#     # Or more simply:
#     matches = dict(zip(row_ind.tolist(), col_ind.tolist()))

#     # Then get the corresponding metrics
#     sensitivities = sensitivities[row_ind, col_ind].tolist()
#     precisions = precisions[row_ind, col_ind].tolist()
#     f1_scores = f1_scores[row_ind, col_ind].tolist()
#     rate_of_agreement = rate_of_agreement[row_ind, col_ind].tolist()
    
#     return matches, rate_of_agreement, f1_scores, sensitivities, precisions

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def spike_matching(dts, dts_pred, fs, old_matches=None, jitter=0.002):
    '''
    For each motor unit, compute the spiking accuracy and validate the matches.
    '''
    precisions = torch.zeros(len(dts_pred), len(dts))
    sensitivities = torch.zeros(len(dts_pred), len(dts))
    rate_of_agreement = torch.zeros(len(dts_pred), len(dts))

    dts = [torch.tensor(dt, dtype=torch.float32) for dt in dts]
    dts_pred = [torch.tensor(dt_pred, dtype=torch.float32) for dt_pred in dts_pred]
    
    jitter_samples = int(jitter * fs)

    for idx, dt_pred in enumerate(dts_pred):
        if dt_pred.numel() == 0:
            continue
        for jdx, dt in enumerate(dts):
            if dt.numel() == 0:
                continue

            time_diffs = torch.abs(dt_pred.reshape(-1, 1) - dt.reshape(1, -1))
            
            # matches_pred = (time_diffs <= jitter_samples).any(dim=1)
            # tps = matches_pred.sum()
            # fps = len(dt_pred) - tps
            
            # matches_true = (time_diffs <= jitter_samples).any(dim=0)
            # fns = (~matches_true).sum()

            matches_true = (time_diffs <= jitter_samples).any(dim=0)
            tps = matches_true.sum()
            fns = len(dt) - tps
            fps = len(dt_pred) - tps


            if (tps + fns) > 0:
                sensitivities[idx, jdx] = tps / (tps + fns)
            if (tps + fps) > 0:
                precisions[idx, jdx] = tps / (tps + fps)
            if (tps + fns + fps) > 0:
                rate_of_agreement[idx, jdx] = tps / (tps + fns + fps)

    if old_matches is None:
        # Use Hungarian algorithm to find the globally optimal assignment
        row_ind, col_ind = linear_sum_assignment((1 - rate_of_agreement).numpy())
    else:
        row_ind, col_ind = np.array(list(old_matches.keys())), np.array(list(old_matches.values()))

    # --- NEW VALIDATION STEP ---
    z_scores = []
    # For each match proposed by the Hungarian algorithm...
    for r, c in zip(row_ind, col_ind):
        # The 'signal' is the RoA of the matched pair
        signal_roa = rate_of_agreement[r, c]

        # The 'noise' distribution is the RoA of this predicted unit
        # with all OTHER ground truth units it was NOT matched with.
        mask = torch.ones(rate_of_agreement.shape[1], dtype=torch.bool)
        mask[c] = False
        noise_roas = rate_of_agreement[r, mask]
        
        # We need at least 2 other units to compute a meaningful std deviation
        if noise_roas.numel() > 1:
            mean_noise = noise_roas.mean()
            std_noise = noise_roas.std()
            # Add a small epsilon to prevent division by zero if all noise RoAs are the same
            z = (signal_roa - mean_noise) / (std_noise + 1e-9)
            z_scores.append(z.item())
        else:
            # If there's only one possible ground truth unit, confidence is maximal.
            # A z-score is not well-defined, so we assign infinity.
            z_scores.append(float('inf'))
    # --- END OF NEW STEP ---

    matches = dict(zip(row_ind.tolist(), col_ind.tolist()))

    matched_sensitivities = sensitivities[row_ind, col_ind].tolist()
    matched_precisions = precisions[row_ind, col_ind].tolist()
    matched_roa = rate_of_agreement[row_ind, col_ind].tolist()
    
    # Calculate F1 scores only for matched pairs to avoid division by zero
    f1_scores = []
    for sens, prec in zip(matched_sensitivities, matched_precisions):
        if (sens + prec) > 0:
            f1_scores.append(2 * sens * prec / (sens + prec))
        else:
            f1_scores.append(0.0)

    return matches, matched_roa, f1_scores, matched_sensitivities, matched_precisions, z_scores

# def spike_matching(dts, dts_pred, fs, jitter=0.002, delay=0):
#     ''' 
#     For each motor unit, compute the spiking accuracy, sensitivity and precision.
#     Searches over possible delays to find the best alignment.
    
#     Args:
#         dts: List of ground truth spike times for each motor unit
#         dts_pred: List of predicted spike times for each motor unit
#         fs: Sampling frequency
#         jitter: Temporal jitter tolerance in seconds
#         delay: Maximum delay to search in seconds (searches from -delay to +delay). 
#                If 0, no delay search is performed.
#     '''
#     print('SPIKE MATCHING...')
#     # Convert lists to tensors
#     dts = [torch.tensor(dt) for dt in dts]
#     dts_pred = [torch.tensor(dt_pred) for dt_pred in dts_pred]
    
#     # Generate delay values to test (in samples, step size = 1 sample)
#     if delay > 0:
#         delay_samples = int(delay * fs)
#         delays = torch.arange(-delay_samples, delay_samples + 1, 1.0)
#         n_delays = len(delays)
#     else:
#         delays = torch.tensor([0.0])
#         n_delays = 1
    
#     precisions = torch.zeros(len(dts_pred), len(dts))
#     sensitivities = torch.zeros(len(dts_pred), len(dts))
#     best_delays = torch.zeros(len(dts_pred), len(dts))
    
#     jitter_samples = int(jitter * fs)
    
#     for idx, dt_pred in tqdm(enumerate(dts_pred)):
#         pred_times = dt_pred.reshape(-1, 1, 1)  # Shape: (n_pred, 1, 1)
        
#         for jdx, dt in enumerate(dts):
#             gt_times = dt.reshape(1, -1, 1)  # Shape: (1, n_true, 1)
#             delays_reshaped = delays.reshape(1, 1, -1)  # Shape: (1, 1, n_delays)
            
#             # Compute time differences for all delays at once
#             # Shape: (n_pred, n_true, n_delays)
#             time_diffs = torch.abs(pred_times - gt_times - delays_reshaped)
            
#             # Check matches for each delay
#             # Shape: (n_pred, n_true, n_delays)
#             within_jitter = time_diffs <= jitter_samples
            
#             # For each delay, compute metrics
#             # True positives: predicted spikes that match any ground truth
#             tps_per_delay = within_jitter.any(dim=1).sum(dim=0)  # Shape: (n_delays,)
            
#             # False negatives: ground truth spikes that don't match any prediction
#             fns_per_delay = (~within_jitter.any(dim=0)).sum(dim=0)  # Shape: (n_delays,)
            
#             # Compute sensitivity and precision for each delay
#             n_pred = len(dt_pred)
#             sensitivities_per_delay = tps_per_delay / (tps_per_delay + fns_per_delay + 1e-12)
#             precisions_per_delay = tps_per_delay / (n_pred + 1e-12)
            
#             # Compute F1 scores for each delay
#             f1_per_delay = 2 * sensitivities_per_delay * precisions_per_delay / \
#                           (sensitivities_per_delay + precisions_per_delay + 1e-12)
            
#             # Find the best delay
#             best_delay_idx = torch.argmax(f1_per_delay)
#             best_delays[idx, jdx] = delays[best_delay_idx]
            
#             # Store metrics for the best delay
#             sensitivities[idx, jdx] = sensitivities_per_delay[best_delay_idx]
#             precisions[idx, jdx] = precisions_per_delay[best_delay_idx]
    
#     # Get best match for each predicted spike with Hungarian algorithm
#     f1_scores = 2 * sensitivities * precisions / (sensitivities + precisions + 1e-12)
#     print('Linear sum assignment...')
#     _, col_ind = linear_sum_assignment((1 - f1_scores).numpy())
    
#     # Vectorize final computations
#     matches = col_ind.tolist()
#     idx_range = torch.arange(len(col_ind))
#     sensitivities = sensitivities[idx_range, col_ind].tolist()
#     precisions = precisions[idx_range, col_ind].tolist()
#     f1_scores = f1_scores[idx_range, col_ind].tolist()
#     best_delays_matched = (best_delays[idx_range, col_ind]).tolist()  # Convert back to seconds
    
#     return matches, f1_scores, sensitivities, precisions, best_delays_matched

def get_muap_correlations(emg_grid, pred_dts, mu_dts, L=50):
    """Computes the correlation between every pair of MUs between session 1 and 2."""
    Npred = len(pred_dts)
    sta_pred = torch.zeros(Npred, 2*L + 1, emg_grid.shape[2], emg_grid.shape[3])
    for idx in range(Npred):
        sta_pred[idx] = get_sta_muaps(emg_grid, pred_dts[idx].astype(int), L=L, plot=False)
    
    Ntrue = len(mu_dts)
    sta_true = torch.zeros(Ntrue, 2*L + 1, emg_grid.shape[2], emg_grid.shape[3])
    for idx in range(Ntrue):
        sta_true[idx] = get_sta_muaps(emg_grid, mu_dts[idx].astype(int), L=L, plot=False)
    
    # Use STA estimates and compute correlation between every pair
    sta_true, sta_pred = sta_true.detach().numpy(), sta_pred.detach().numpy()
    corrs = np.zeros((Npred, Ntrue))
    for idx in range(Npred):
        for jdx in range(Ntrue):
            corr, p = scipy.stats.pearsonr(sta_true[jdx].flatten(), sta_pred[idx].flatten())
            corrs[idx, jdx] = corr
    return corrs


def compute_mu_crosscorrelation(mu_set1, mu_set2, duration, fs=2048, 
                                jitter=0.002, max_lag=0.5, device='cpu'):
    """
    Compute temporal cross-correlations between motor unit pairs from two sets.
    Vectorized implementation using PyTorch for efficiency.
    
    Parameters:
    -----------
    mu_set1 : list of arrays
        List of arrays, each containing discharge times (in seconds) for one MU
    mu_set2 : list of arrays
        List of arrays, each containing discharge times (in seconds) for one MU
    duration : float
        Total duration of recording in seconds
    fs : int, optional
        Sampling frequency in Hz (default: 2048)
    window_size : float, optional
        Moving average window size in seconds (default: 0.4)
    max_lag : float, optional
        Maximum lag for cross-correlation in seconds (default: 0.5)
    device : str, optional
        Device to run computations on ('cpu' or 'cuda', default: 'cpu')
    
    Returns:
    --------
    xcorr_matrix : ndarray
        Cross-correlation matrix of shape (Nmu1, Nmu2, 2*max_lag_samples+1)
    lags : ndarray
        Time lags corresponding to cross-correlation values (in seconds)
    """
    
    # Convert duration and parameters to samples
    T = int(duration * fs)
    window_samples = 2*int(jitter*fs) + 1
    max_lag_samples = int(max_lag * fs)
    
    Nmu1 = len(mu_set1)
    Nmu2 = len(mu_set2)
    
    # Create spike trains for set 1
    spike_trains1 = np.zeros((Nmu1, T))
    for i, discharge_times in enumerate(mu_set1):
        spike_indices = (discharge_times * fs).astype(int)
        spike_indices = spike_indices[spike_indices < T]
        spike_trains1[i, spike_indices] = 1
    
    # Create spike trains for set 2
    spike_trains2 = np.zeros((Nmu2, T))
    for i, discharge_times in enumerate(mu_set2):
        spike_indices = (discharge_times * fs).astype(int)
        spike_indices = spike_indices[spike_indices < T]
        spike_trains2[i, spike_indices] = 1
    
    # Convert to PyTorch tensors
    spike_trains1 = torch.from_numpy(spike_trains1).float().to(device)
    spike_trains2 = torch.from_numpy(spike_trains2).float().to(device)
    
    # Apply moving average filter to set 1 using 1D convolution
    # Shape for conv1d: (batch, channels, length) = (1, Nmu1, T)
    spike_trains1 = spike_trains1.unsqueeze(0)  # (1, Nmu1, T)
    
    # Create moving average kernel
    window = torch.ones(Nmu1, 1, window_samples, device=device)
    
    # Apply depthwise convolution (each channel filtered independently)
    padding = window_samples // 2  # For centered convolution
    filtered_trains1 = torch.nn.functional.conv1d(spike_trains1, window, 
                                padding=padding, groups=Nmu1)
    
    # Trim to original length if needed (for even window sizes)
    if filtered_trains1.shape[2] > T:
        filtered_trains1 = filtered_trains1[:, :, :T]
    
    filtered_trains1 = filtered_trains1.squeeze(0)  # (Nmu1, T)
    
    # Compute cross-correlations using FFT-based convolution
    # Pad signals for full cross-correlation
    pad_len = T - 1
    filtered_trains1_padded = torch.nn.functional.pad(filtered_trains1, (pad_len, pad_len))
    
    # Flip spike_trains2 for cross-correlation (correlate = convolve with flipped signal)
    spike_trains2_flipped = torch.flip(spike_trains2, dims=[1])
    
    # Reshape for batch convolution
    # filtered_trains1: (Nmu1, 1, T_padded)
    # spike_trains2_flipped: (Nmu2, 1, T)
    filtered_trains1_padded = filtered_trains1_padded.unsqueeze(1)  # (Nmu1, 1, T_padded)
    spike_trains2_flipped = spike_trains2_flipped.unsqueeze(1)  # (Nmu2, 1, T)
    
    # Compute all cross-correlations at once using grouped convolution
    # For each MU in set 1, correlate with all MUs in set 2
    xcorr_full = []
    for i in range(Nmu1):
        # Convolve one signal from set1 with all signals from set2
        xcorr_i = torch.nn.functional.conv1d(filtered_trains1_padded[i:i+1].expand(Nmu2, 1, -1),
                          spike_trains2_flipped,
                          groups=Nmu2)
        xcorr_full.append(xcorr_i.squeeze(1))  # (Nmu2, corr_length)
    
    xcorr_full = torch.stack(xcorr_full, dim=0)  # (Nmu1, Nmu2, corr_length)
    
    # Extract relevant lag range
    center = xcorr_full.shape[2] // 2
    xcorr_matrix = xcorr_full[:, :, center - max_lag_samples:center + max_lag_samples + 1]
    
    # Convert back to numpy
    xcorr_matrix = xcorr_matrix.cpu().numpy()
    
    # Create lag vector in seconds
    lags = np.arange(-max_lag_samples, max_lag_samples + 1) / fs
    
    return xcorr_matrix, lags


def get_aligned_discharge_times(emg_grid, discharge_times_list, L, energy_threshold=0.2):
    """
    Align discharge times for multiple MUs based on cumulative energy criterion.
    
    Parameters:
    -----------
    emg_grid : torch.Tensor, shape (T, 1, H, W)
        EMG grid data
    discharge_times_list : list of numpy arrays
        List where each element is a numpy array of discharge times for one MU
    L : int
        MUAP window length
    energy_threshold : float
        Fraction of energy that should occur before discharge time (default 0.01 for 99% after)
    
    Returns:
    --------
    aligned_discharge_times_list : list of numpy arrays
        Aligned discharge times for each MU
    shifts : list of int
        Shift applied to each MU (for debugging/analysis)
    """
    aligned_discharge_times_list = []
    shifts = []
    
    for mu_idx, discharge_times in enumerate(discharge_times_list):
        # Convert to torch tensor for processing
        discharge_times_torch = torch.tensor(discharge_times, device=emg_grid.device).to(torch.int)
        
        # Get STA MUAP template for this MU
        sta_muap = get_sta_muaps(emg_grid, discharge_times_torch, L, plot=False)  # Shape: (L, H, W)
        
        # Compute energy per time sample across all spatial channels
        energy_per_sample = torch.sum(sta_muap**2, dim=(1, 2))  # Shape: (L,)
        
        # Cumulative energy
        cumulative_energy = torch.cumsum(energy_per_sample, dim=0)
        total_energy = cumulative_energy[-1]
        
        # Find the time index where cumulative energy reaches the threshold
        threshold_energy = energy_threshold * total_energy
        onset_idx = torch.argmax((cumulative_energy >= threshold_energy).float())
        
        # Current discharge time is at the center (L // 2)
        current_center = L // 2
        
        # Shift needed: move discharge time from current_center to onset_idx
        shift = onset_idx.item() - current_center
        
        # Apply shift and convert back to numpy
        aligned_times = discharge_times + shift

        # Filter out times that are out of bounds
        # Keep only: 0 <= aligned_time < T
        valid_mask = (aligned_times >= 0) & (aligned_times < emg_grid.shape[0])
        aligned_times_valid = aligned_times[valid_mask]
        
        assert np.max(aligned_times_valid) < emg_grid.shape[0] and np.max(aligned_times_valid) >= 0

        aligned_discharge_times_list.append(aligned_times_valid)
        shifts.append(shift)
    
    return aligned_discharge_times_list, shifts


def out_of_bounds_pixels(height: int, width: int, theta: float):
    """
    Computes the number of out-of-bound pixels along the horizontal and vertical directions
    after rotating an image by an angle theta.
    
    Parameters:
        height (int): Height of the original image.
        width (int): Width of the original image.
        theta (float): Rotation angle in radians.
    
    Returns:
        (float, float): Tuple containing out-of-bounds pixels (delta_width, delta_height)
    """
    
    # Compute new bounding box dimensions
    new_width = abs(width * np.cos(theta)) + abs(height * np.sin(theta))
    new_height = abs(width * np.sin(theta)) + abs(height * np.cos(theta))
    
    # Compute out-of-bounds pixels
    y_margin = (new_width - width) / 2
    x_margin = (new_height - height) / 2
    
    return y_margin, x_margin

# def handle_outliers(emg_grid):
#     '''Determine outlier channels based on spectral flatness, and replace them with average of neighbours.'''
#     # Determine coordinates of outliers
#     H, W = emg_grid.shape[2:]
#     flatness = torch.tensor(get_spectral_flatness_ar2(emg_grid.squeeze()))
#     Q1, Q3 = torch.quantile(flatness.flatten(), torch.tensor([0.25, 0.75]))
#     IQR = Q3 - Q1
#     upper = Q3 + 1.5*IQR
#     y, x = torch.where(flatness >= upper) # only keep non-noisy channel
#     y, x = y.tolist(), x.tolist()

#     idx = 0
#     while idx < len(y): # for each outlier
#         l,r,b,t = x[idx] != 0, x[idx] != W-1, y[idx] != H-1, y[idx] != 0
#         subgrid = emg_grid[:, :, y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten(start_dim=2, end_dim=3)
#         subgrid_flatness = flatness[y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten()
#         subgrid = subgrid[:, :, subgrid_flatness < upper] # remove outlier channels included
#         if subgrid.shape[2] < 3: # if less than 3 valid neighbours, try again after filling in more channels
#             y.append(y[idx])
#             x.append(x[idx])
#         else:
#             emg_grid[:,:,y[idx], x[idx]] = subgrid.mean(dim=2) # compute as average of neighbours
#         idx += 1

#     return emg_grid

def handle_outliers(emg_grid, visible_outliers=None):
    '''Determine outlier channels, and replace them with average of neighbours, 
       prioritizing those with the fewest outlier neighbours.'''

    emg_grid = emg_grid.numpy()
    H, W = emg_grid.shape[2:]

    flatness = get_spectral_flatness_ar2(emg_grid.squeeze())
    Q1, Q3 = np.quantile(flatness.flatten(), [0.25, 0.75])
    IQR = Q3 - Q1
    upper = Q3 + IQR
    outlier_mask = flatness >= upper  # boolean mask of outliers

    if visible_outliers is not None:
        outlier_mask = np.logical_or(outlier_mask, visible_outliers)

    # 1. Count outlier neighbors for each pixel
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count_img = signal.convolve2d(outlier_mask.astype(np.uint8), kernel, mode='same', boundary='fill', fillvalue=0)

    # 2. Add 3 to all edge pixels
    neighbor_count_img[0, :] += 3
    neighbor_count_img[-1, :] += 3
    neighbor_count_img[:, 0] += 3
    neighbor_count_img[:, -1] += 3

    # 3. Add 2 more to corners
    neighbor_count_img[0, 0] += 2
    neighbor_count_img[0, -1] += 2
    neighbor_count_img[-1, 0] += 2
    neighbor_count_img[-1, -1] += 2

    # 4. Get outlier indices and their neighbor counts
    outlier_indices = np.argwhere(outlier_mask)
    neighbor_counts = neighbor_count_img[outlier_mask]

    # 5. Sort outliers by neighbor count (ascending)
    sorted_indices = np.argsort(neighbor_counts)
    outlier_indices = outlier_indices[sorted_indices]

    # 6. Fill outliers with mean of valid neighbors
    for y, x in outlier_indices:
        y0, y1 = max(0, y-1), min(H, y+2)
        x0, x1 = max(0, x-1), min(W, x+2)
        neighbor_mask = ~outlier_mask[y0:y1, x0:x1].copy()
        center_rel_y = y - y0
        center_rel_x = x - x0
        neighbor_mask[center_rel_y, center_rel_x] = False  # exclude center
        neighbors = emg_grid[..., y0:y1, x0:x1][..., neighbor_mask]
        if neighbors.size > 0:
            emg_grid[..., y, x] = neighbors.mean(axis=-1)
        # else: leave as is

    return torch.tensor(emg_grid)

# def handle_outliers_old(emg_grid):
#     '''Determine outlier channels, and replace them with average of neighbours.'''
#     # Determine coordinates of outliers
#     H, W = emg_grid.shape[2:]
#     emg_grid_var = emg_grid.var(dim=[0,1])
#     Q1, Q3 = torch.quantile(emg_grid_var.flatten(), 0.25), torch.quantile(emg_grid_var.flatten(), 0.75)
#     IQR = Q3 - Q1
#     lower, upper = Q1 -3.0*IQR, Q3 + 3.0*IQR
#     y, x = torch.where(torch.logical_or(emg_grid_var >= upper, emg_grid_var <= lower)) # only keep non-noisy channel
#     y, x = y.tolist(), x.tolist()

#     idx = 0
#     while idx < len(y): # for each outlier
#         l,r,b,t = x[idx] != 0, x[idx] != W-1, y[idx] != H-1, y[idx] != 0
#         subgrid = emg_grid[:, :, y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten(start_dim=2, end_dim=3)
#         subgridvar = emg_grid_var[y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten()
#         subgrid = subgrid[:, :, torch.logical_and(subgridvar < upper, subgridvar > lower)] # remove outlier channels included
#         if subgrid.shape[2] < 3: # if less than 3 valid neighbours, try again after filling in more channels
#             y.append(y[idx])
#             x.append(x[idx])
#         else:
#             emg_grid[:,:,y[idx], x[idx]] = subgrid.mean(dim=2) # compute as average of neighbours
#         idx += 1

#     return emg_grid

def get_min_distance(grid_shape, Tx, Ty, theta):
    '''Obtain minimum distance of a given electrode in transformed grid to an electrode in the old grid coordiantes, averaged across electrodes.'''
    H, W = grid_shape
    original_grid = get_transformed_grid((1, 1, H, W))
    transformed_grid = get_transformed_grid((1, 1, H, W), Tx, Ty, theta)
    center = transformed_grid[0:1, H//2:H//2 + 1, W//2:W//2 + 1, :]
    distances = torch.linalg.norm(original_grid - center, dim=3)
    min_distance = torch.min(distances)
    return original_grid, transformed_grid, min_distance

def get_min_conservative_crop(grid_shape, transformed_grid, original_grid):
    '''Given a transformed grid and original grid coordinates, find the smallest crop for each side such that no dead channels are included.'''
    H, W = grid_shape
    transformed_coordinates = transformed_grid[0, :, :, :].cpu().numpy().reshape(-1, 2)
    original_coordinates = original_grid[0, :, :, :].cpu().numpy().reshape(-1, 2)
    hull = ConvexHull(transformed_coordinates)
    delaunay = Delaunay(transformed_coordinates[hull.vertices])
    inside = delaunay.find_simplex(original_coordinates) >= 0
    mask = torch.tensor(inside.reshape(H, W))
    
    # Find most conservative crop
    lcrop, rcrop, bcrop, tcrop = W//2 - 1, W//2 - 1, H//2 - 1, H//2 - 1
    min_crop = False
    while not min_crop:
        crop_sum = lcrop + rcrop + bcrop + tcrop
        if mask[tcrop:H-bcrop, lcrop-1:W-rcrop].all() and lcrop > 0:
            lcrop -= 1
        if mask[tcrop:H-bcrop, lcrop:W-(rcrop-1)].all() and rcrop > 0:
            rcrop -= 1
        if mask[tcrop-1:H-bcrop, lcrop:W-rcrop].all() and tcrop > 0:
            tcrop -= 1
        if mask[tcrop:H-(bcrop-1), lcrop:W-rcrop].all() and bcrop > 0:
            bcrop -= 1
        if crop_sum == lcrop + rcrop + bcrop + tcrop: # if no more changes, we have found the minimum crop
            min_crop = True
    return lcrop, rcrop, bcrop, tcrop


def refine_sep_mat(emg_grid_transform, sda, base_loss=1.00, nepochs=50, batch_size=2048, lr=1e-4, device='cpu', loss='kurtosis', R=16):
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
    # engine = scipy.stats.qmc.LatinHypercube(d=5)
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
            sda.sal.xshift[0].data, sda.sal.yshift[0].data, sda.sal.rot_theta[0].data = init_params[npoint, :3]
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

    return losses