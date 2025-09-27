import torch
import scipy
import numpy as np
from loss_functions import KurtosisLoss, NegentropyLoss
from tqdm import tqdm


def generate_gaussian_muaps(mu_count, H, W, L, fxmax, sampfactor=10):
    '''Generates gaussian sequence with a sampling rate freqfactor times greater than desired, so that we can lowpass and then downsample.
    '''
    muaps = np.random.normal(size=(mu_count, H*sampfactor, W*sampfactor, L*sampfactor)) # create oversampled MUAP so that digital filtering is more appropriate

    # Lowpass temporally to nyquist temporal frequency and downsample temporally to save computational overhead
    b, a = scipy.signal.butter(N=2, Wn=0.5/sampfactor, btype='low') # normalized temporal frequency of a quarter of the sampling rate
    muaps = scipy.signal.filtfilt(b, a, muaps, axis=3) # filter along rows
    muaps = muaps[:,:,:,::sampfactor] # downsample after temporal lowpass

    # Lowpass signals to introduce smoothness
    b,a = scipy.signal.butter(N=2, Wn=fxmax/sampfactor, btype='low')
    muaps = scipy.signal.filtfilt(b, a, muaps, axis=1) # filter along rows
    muaps = scipy.signal.filtfilt(b, a, muaps, axis=2) # filter along columns
    return torch.tensor(muaps.copy()).to(torch.float32)

def generate_spike_trains(mu_count, duration, Tmean=60, ISV=0.15):
    ''' Generate motor unit spike trains, both at very high as well as normal resolution.'''
    
    spts = torch.zeros(mu_count, duration)
    dts = []
    Tstd = int(Tmean*ISV) # ISV relates mean to STD
    # For every independent MU
    for mu_idx in range(mu_count):
        times = torch.normal(mean=torch.ones(int(duration/Tmean))*Tmean, std=Tstd)
        times = torch.cumsum(times, dim=0)#.to(torch.int64) # cumulative sum of firing times
        int_times = torch.round(times).to(int)
        fr_times = int_times[int_times < duration]
        dts.append(fr_times) # become discharge times in seconds
        spts[mu_idx, fr_times] = 1 # set all firing time values to 1

    return spts, dts

def generate_emg(spts, muaps, device='cpu'):
    ''' Generate EMG based on spike trains and simulated MUAPs.'''
    # Pad input spike trains
    spts = spts.view(1, spts.shape[0], 1, spts.shape[1]) # reshape for convolution input
    padding_total = muaps.shape[-1] - 1 # muap/symbol length
    # right_padding = (padding_total + R) // 2 # account for R in padding so that we can use middle segment of MUAP in separation vectors
    # left_padding = padding_total - right_padding
    # spts = torch.nn.functional.pad(spts, (left_padding, right_padding), mode='constant', value=0)
    spts = torch.nn.functional.pad(spts, (padding_total, 0), mode='constant', value=0) # for causal convolution

    # Prepare muaps as kernels
    H, W = muaps.shape[1], muaps.shape[2]
    muaps = muaps.reshape(muaps.shape[0], muaps.shape[1]*muaps.shape[2], muaps.shape[3]) # flatten over channels
    muaps = torch.transpose(muaps, dim0=0, dim1=1).unsqueeze(2)
    muaps = torch.flip(muaps, dims=[3]) # flip kernel to obtain proper convolution 
    EMG = torch.nn.functional.conv2d(spts.to(device), muaps.to(device)) # obtain convolutive mixture
    EMG = EMG.squeeze().view(H, W, EMG.shape[-1]) # reshape back into EMG grid
    return EMG

def downsample_muaps(muaps, sampfactor=10):
    '''Downsample the MUAPs before generating separation vectors.'''
    return muaps[:, ::sampfactor, ::sampfactor, :] # downsample muaps along spatial coordinates

def get_sta_templates(muaps, R=None, delay=None, xcrop=0, ycrop=0):
    ''' Based on MUAPs, just generate the separation vectors neccessary.'''
    R = R if R is not None else muaps.shape[-1]
    N, H, W, L = muaps.shape
    # delay = torch.ceil(torch.tensor(((L + R) / 2))).to(torch.int) - 1 # delay introduced by causality of triggering process
    # delay = L - 1
    if delay is None:
        delay = (torch.floor(torch.tensor([L + R]/2)) - 1) .to(torch.int) # delay introduced by causality of triggering process
    else:
        delay = delay
    
    if R is None: R = L
    Nch = (H-2*ycrop)*(W-2*xcrop)
    B = torch.zeros(N, Nch*R)
    for mdx in range(N):
        for l in range(R):
            B[mdx, l*Nch:(l+1)*Nch] = muaps[mdx, ycrop:H-ycrop, xcrop:W-xcrop, delay-l].ravel() # MUAP reversed is the separation vector itself!
        B[mdx, :] = B[mdx, :] / (torch.linalg.vector_norm(B[mdx, :]) + 1e-12) # make a unit vector
    return B

def add_noise(emg, SNR):
    '''Adding white noise given a particular SNR ratio (dB) for EMG. Since it's a simple operation,
        we move it to CPU to avoid GPU memory overhead.
    '''
    device = emg.device
    emg_cpu = emg.to('cpu')
    var_signal = torch.var(emg_cpu)
    var_noise = var_signal / (10**(SNR/10)) # noise variance to be injected
    emg_cpu = emg_cpu + torch.randn_like(emg_cpu)*torch.sqrt(var_noise) # add noise
    return emg_cpu.to(device)

def simulation_make_grid(emg):
    '''Function that converts EMG grid into the dimensions of a batch of images, expected by the affine transforms and decomposition module. (Input shape H, W, T)'''
    H, W = emg.shape[0], emg.shape[1]
    emg = emg.reshape(-1, emg.shape[2])
    emg_grid = emg.T.reshape((emg.shape[1], 1) + (H, W))
    return emg_grid

def downsample_grid(emg_grid, sampfactor=10):
    '''Downsample EMG signal given the originally used sampfactor.'''
    return emg_grid[:, :, ::sampfactor, ::sampfactor] # downsamples EMG grid spatially

# def search_fit_sda(emg_grid_transform, sda, base_loss, npoints=50, nepochs=50, lr=1e-4, device='cpu', loss='kurtosis', frozen_sep_mat=True, plot=0):
#     ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''    
#     _, _, H, W = emg_grid_transform.shape
#     if loss == 'kurtosis':
#         ica_loss = KurtosisLoss()
#     else:
#         ica_loss = NegentropyLoss()
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sda.parameters()),
#                                                 lr=lr)                 

#     # Collect output tensors
#     output_list = []
#     losses = []
#     xshifts,yshifts,angles,xscales,yscales = [], [], [], [], []

#     # Freeze all parameters
#     for param in sda.parameters():
#         param.requires_grad = False
    
#     # Searching through initial conditions
#     print('SAMPLING AND EVALUATING INITIAL CONDITIONS...')
#     boundaries = torch.tensor([3.0, 3.0, 20*np.pi/180, 0.2, 0.2]) # symmetric for each dimension about zero
#     losses = torch.zeros(npoints)
#     engine = scipy.stats.qmc.LatinHypercube(d=5)
#     init_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float32)-1 # scale from [0,1] to [-1, 1]
#     init_params[:,0], init_params[:,1], init_params[:, 2] = 2*boundaries[0]*init_params[:,0]/(W-1), 2*boundaries[1]*init_params[:,1]/(H-1), boundaries[2]*init_params[:, 2]/np.pi
#     init_params[:, 3] = torch.pow((1 + torch.abs(init_params[:, 3])*boundaries[3]), torch.sign(init_params[:, 3]) ) # generates scalings appropriately
#     init_params[:, 4] = torch.pow((1 + torch.abs(init_params[:, 4])*boundaries[4]), torch.sign(init_params[:, 4]) )

#     init_params = init_params.to(device)
#     sda.train() # leave batch norm parameters adaptive
#     with torch.no_grad():
#         for npoint in tqdm(range(npoints)):
#             emg_grid_copy = emg_grid_transform.clone().detach()
#             # Set initial conditions
#             # sda.sal.xshift[0].data, sda.sal.yshift[0].data, sda.sal.rot_theta[0].data = init_params[npoint, :3]
#             # sda.sal.xscale[0].data, sda.sal.yscale[0].data = init_params[npoint, 3:]
#             sda.sal.xshift[0].copy_(init_params[npoint, 0])
#             sda.sal.yshift[0].copy_(init_params[npoint, 1])
#             sda.sal.rot_theta[0].copy_(init_params[npoint, 2])
#             sda.sal.xscale[0].copy_(init_params[npoint, 3])
#             sda.sal.yscale[0].copy_(init_params[npoint, 4])
#             # Evaluate loss function at given condition
#             outputs = sda(emg_grid_copy.to(device)).to(device)  # Shape will be (batch_size, num_classes)

#             # Compute ICA Loss and backprop    
#             loss = ica_loss(outputs)
#             losses[npoint] = loss.item()

#         if npoints > 0:
#             losses = losses / base_loss # normalize by baseline loss
#             # sda.sal.xshift[0].data, sda.sal.yshift[0].data, sda.sal.rot_theta[0].data = init_params[losses.argmax(), :3] # get best initialization
#             # sda.sal.xscale[0].data, sda.sal.yscale[0].data = init_params[losses.argmax(), 3:]
#             sda.sal.xshift[0].copy_(init_params[losses.argmax(), 0]) # get best initialization
#             sda.sal.yshift[0].copy_(init_params[losses.argmax(), 1])
#             sda.sal.rot_theta[0].copy_(init_params[losses.argmax(), 2])
#             sda.sal.xscale[0].copy_(init_params[losses.argmax(), 3])
#             sda.sal.yscale[0].copy_(init_params[losses.argmax(), 4])
            
#             print(f'TOP 5 LOSS VALUES SAMPLED: {torch.topk(losses, k=torch.min(torch.tensor([npoints, 5])))}')

#     # Make SAL parameters learnable
#     # for param in sda.sal.parameters():
#     if frozen_sep_mat:
#         for param in sda.sal.parameters():
#             param.requires_grad = True        
#     else:
#         for param in sda.parameters():
#             param.requires_grad = True

#     # Loop through the DataLoader
#     print('TRAINING FROM BEST INIT. CONDITION...')
#     losses = []
#     for ne in tqdm(range(nepochs)):
#         # Forward pass through the model
#         outputs = sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

#         # Compute ICA Loss and backprop    
#         loss = ica_loss(outputs)
#         optimizer.zero_grad()
#         loss.backward()
#         print('LOSS:', loss.item()/base_loss)
#         optimizer.step()
#         print(f'PARAMS:\n xshift: {(W-1)*sda.sal.xshift[0].item()/2}, yshift: {(H-1)*sda.sal.yshift[0].item()/2}, theta: {sda.sal.rot_theta[0].item()} ')
#         print(f'xscale: {sda.sal.xscale[0].item()}, yscale: {sda.sal.yscale[0].item()}')
#         # Collect outputs and loss
#         output_list.append(outputs)
#         losses.append(loss.item())
#         xshifts.append(sda.sal.xshift[0].item())
#         yshifts.append(sda.sal.yshift[0].item())
#         angles.append(sda.sal.rot_theta[0].item())
#         xscales.append(sda.sal.xscale[0].item())
#         yscales.append(sda.sal.yscale[0].item())

#     # if plot:
#     #     fig, axs = plt.subplots(1, 2)
#     #     axs[0].plot(losses)
#     #     axs[0].set_title('Training Loss')
#     #     axs[1].plot(W*(np.array(xshifts).ravel())/2)
#     #     axs[1].plot(H*(np.array(yshifts).ravel())/2)
#     #     axs[1].plot(angles)
#     #     axs[1].plot(xscales)
#     #     axs[1].plot(yscales)
#     #     axs[1].hlines(y=[-params['Tx'], -params['Ty'], -params['theta'], 1/params['xscale'], 1/params['yscale']],
#     #                     xmin=0, xmax=len(np.array(xshifts).ravel()), linestyles='dashed', label='ground truth')
#     #     axs[1].legend(['xshift-pred','yshift-pred', 'theta', 'xscale', 'yscale'])
#     #     axs[1].set_title('Parameter Dynamics')
#     #     axs[1].set_ylim([-3.0, 3.0])
#     #     plt.savefig('learning.jpg')

#     # Get final outputs, i.e. optimal souces
#     with torch.no_grad():
#         final_outputs = sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

#     sources = final_outputs.detach().cpu()
#     return sources, losses