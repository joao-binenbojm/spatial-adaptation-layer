import numpy as np
from math import ceil
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import seaborn as sns

# Bayesian Optimization code
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.kernels import RBFKernel
from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
# from botorch.distributions import LogNormal


from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg
from loss_functions import KurtosisLoss, NegentropyLoss
from sda import SpatialDecompositionAdaptation

def inject_gradient_noise(model, epoch, total_epochs, initial_noise_std=0.1):
    """
    Gradually reduce the noise standard deviation as training progresses.
    Args:
        model (torch.nn.Module): The model to modify gradients for.
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        initial_noise_std (float): Initial standard deviation of noise.
    """
    noise_std = initial_noise_std * (1 - epoch / total_epochs)  # Decrease noise as training progresses
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad += noise  # Add the noise to the gradient

# Reflective boundary function
def apply_reflective_boundary(param, bound):
    if param < -bound:  # Below lower bound
        param = bound + (bound - param)
    elif param > bound:  # Above upper bound
        param = bound - (param - bound)
    return param

class SDAExperiment:

    def __init__(self):
        self.params = {}

    def generate_gaussian_muaps(self, mu_count, H, W, L, fxmax, sampfactor=100):
        '''Generates gaussian sequence with a sampling rate freqfactor times greater than desired, so that we can lowpass and then downsample.
        '''
        params = {'mu_count': mu_count, 'H': H, 'W': W, 'L':L, 'fxmax':fxmax, 'sampfactor':sampfactor}
        self.params.update(params)# keep track of chosing experimental parameters

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

    def generate_spike_trains(self, mu_count, duration, Tmean=60, ISV=0.15):
        ''' Generate motor unit spike trains, both at very high as well as normal resolution.'''
        params = {'mu_count': mu_count, 'duration': duration, 'sampfactor': sampfactor, 'Tmean':Tmean, 'ISV':ISV}
        self.params.update(params)# keep track of chosing experimental parameters
        
        spts = torch.zeros(mu_count, duration)
        dts = []
        Tstd = int(Tmean*ISV) # ISV relates mean to STD
        # For every independent MU
        for mu_idx in range(mu_count):
            times = torch.normal(mean=torch.ones(int(duration/Tmean))*Tmean, std=Tstd)
            times = torch.cumsum(times, dim=0)#.to(torch.int64) # cumulative sum of firing times
            int_times = torch.round(times).to(torch.int64)
            dts.append(int_times) # become discharge times in seconds
            fr_times = int_times[int_times < duration]
            spts[mu_idx, fr_times] = 1 # set all firing time values to 1

        return spts, dts

    # def generate_emg(self, spts, muaps):
    #     ''' Generate EMG based on spike trains and simulated MUAPs.'''
    #     EMG = torch.zeros(muaps.shape[1], muaps.shape[2], spts.shape[1]) # empty EMG array
    #     for mdx in tqdm(range(muaps.shape[0])):
    #         for row in range(muaps.shape[1]):
    #             for col in range(muaps.shape[2]):
    #                 EMG[row, col, :] += convolve(spts[mdx,:], muaps[mdx, row, col, :], mode='same')
    #     return EMG

    def generate_emg(self, spts, muaps, device='cpu'):
        ''' Generate EMG based on spike trains and simulated MUAPs.'''
        # Pad input spike trains
        spts = spts.view(1, spts.shape[0], 1, spts.shape[1]) # reshape for convolution input
        padding_total = muaps.shape[-1] - 1 # muap/symbol length
        left_padding = padding_total // 2    
        right_padding = padding_total - left_padding
        spts = torch.nn.functional.pad(spts, (left_padding, right_padding), mode='constant', value=0)

        # Prepare muaps as kernels
        H, W = muaps.shape[1], muaps.shape[2]
        muaps = muaps.view(muaps.shape[0], muaps.shape[1]*muaps.shape[2], muaps.shape[3]) # flatten over channels
        muaps = torch.transpose(muaps, dim0=0, dim1=1).unsqueeze(2)
        muaps = torch.flip(muaps, dims=[3]) # flip kernel to obtain proper convolution 
        EMG = torch.nn.functional.conv2d(spts.to(device), muaps.to(device)) # obtain convolutive mixture
        EMG = EMG.squeeze().view(H, W, EMG.shape[-1]) # reshape back into EMG grid
        return EMG

    def add_noise(self, emg, SNR):
        '''Adding white noise given a particular SNR ratio (dB) for EMG.'''
        self.params.update({'SNR': SNR})
        var_signal = torch.var(emg)
        var_noise = var_signal / (10**(SNR/10)) # noise variance to be injected
        emg = emg + torch.randn_like(emg)*torch.sqrt(var_noise) # add noise
        return emg

    def make_grid(self, emg):
        '''Function that converts EMG grid into the dimensions of a batch of images, expected by the affine transforms and decomposition module. (Input shape H, W, T)'''
        H, W = emg.shape[0], emg.shape[1]
        emg = emg.reshape(-1, emg.shape[2])
        emg_grid = emg.T.reshape((emg.shape[1], 1) + (H, W))
        return emg_grid

    def apply_affine(self, emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, sampfactor=100):
        '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''
        params = {'Tx': Tx, 'Ty': Ty, 'theta': theta, 'xscale':xscale, 'yscale':yscale, 'sampfactor': sampfactor}
        self.params.update(params)# keep track of chosing experimental parameters

        N, C, H, W = emg_grid.shape
        Tx, Ty = torch.tensor(2*Tx*sampfactor/W), torch.tensor(2*sampfactor*Ty/H) # Normalize translation values automatically
        theta, xscale, yscale = torch.tensor(theta), torch.tensor(xscale), torch.tensor(yscale)

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

    def downsample_grid(self, emg_grid, sampfactor=100):
        '''Downsample EMG signal given the originally used sampfactor.'''
        return emg_grid[:, :, ::sampfactor, ::sampfactor] # downsamples EMG grid spatially

    def downsample_muaps(self, muaps, sampfactor=100):
        '''Downsample the MUAPs before generating separation vectors.'''
        return muaps[:, ::sampfactor, ::sampfactor, :] # downsample muaps along spatial coordinates

    def grid_crop(self, emg_grid, xcrop=0, ycrop=0):
        ''' Keep only a subgrid at the center, returning a signal of shape (H - 2ycrop, W - 2xcrop)'''
        cropped_emg = emg_grid.clone() # ensures no aliasing issues
        cropped_emg = cropped_emg[:, :, ycrop:cropped_emg.shape[2]-ycrop, xcrop:cropped_emg.shape[3]-xcrop]
        return cropped_emg

    def get_separation_vectors(self, muaps, R=None, xcrop=0, ycrop=0):
        ''' Based on MUAPs, just generate the separation vectors neccessary.'''
        R = R if R is not None else muaps.shape[-1]
        params = {'R': R, 'xcrop': xcrop, 'ycrop': ycrop}
        self.params.update(params)# keep track of chosing experimental parameters

        N, H, W, L = muaps.shape
        if R is None: R = L
        Nch = (H-2*ycrop)*(W-2*xcrop)
        B = torch.zeros(N, Nch*R)
        for mdx in range(N):
            for l in range(R):
                B[mdx, l*Nch:(l+1)*Nch] = muaps[mdx, ycrop:H-ycrop, xcrop:W-xcrop, R-l].ravel() # MUAP reversed is the separation vector itself!
            B[mdx, :] = B[mdx, :] / (torch.norm(B[mdx, :]) + 1e-12) # make a unit vector
        return B
    
    def get_whiten_matrix(self, emg_grid, B):
        ''' Get whiten matrix based on EMG and apply transpose to separation vectors.'''
        emg = np.array(emg_grid).squeeze().reshape(emg_grid.shape[0], -1).T
        extended_emg_template = np.zeros((self.params['R']*emg.shape[0], emg.shape[1] + self.params['R'] - 1))
        extended_emg = extend_emg(extended_emg_template, emg, self.params['R'])
        _,self.whiten_mat,_ = whiten_emg(extended_emg) # we don't care about the whitened emg for now
        self.whiten_mat = torch.tensor(self.whiten_mat).to(torch.float32)
        self.sep_mat = B @ self.whiten_mat.T # multiply by transpose of whiten matrix (i.e. inverse since orthogonal)
        self.whiten_mat.requires_grad, self.sep_mat.requires_grad = True, True
        return self.sep_mat @ self.whiten_mat @ torch.tensor(extended_emg).to(torch.float32) # return sources

    def get_base_loss(self, emg_grid, loss='kurtosis', device='cpu'):
        '''Getting base loss.'''
        N, C, H, W = emg_grid.shape
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        with torch.no_grad():
            self.base_loss = ica_loss(sda(emg_grid.to(device))).item()


    def fit_sda(self, emg_grid_transform, nepochs=100, lr=1e-4, device='cpu', loss='kurtosis', plot=1):
        ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''
        params = {'nepochs': nepochs, 'lr': lr}
        self.params.update(params)# keep track of chosing experimental parameter
        
        N, C, H, W = emg_grid_transform.shape
        self.sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.sda.parameters()),
        #                                             lr=lr)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.sda.parameters()),
                                                    lr=lr)

        # Collect output tensors
        output_list = []
        losses = []
        xshifts,yshifts,angles = [], [], []

        # Freeze all parameters except for SAL parameters
        for param in self.sda.parameters():
            param.requires_grad = False
        
        # for param in self.sda.sal.parameters():
            # param.requires_grad = True
        self.sda.sal.xshift.requires_grad = True
        self.sda.sal.yshift.requires_grad = True
        self.sda.sal.rot_theta.requires_grad = True
        # base_noise_scale = 0.3

        for ne in tqdm(range(nepochs)):
            # Forward pass through the model
            outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

            # Compute ICA Loss and backprop    
            loss = ica_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            print('LOSS:', loss.item()/self.base_loss)

            optimizer.step()
            print(f'PARAMS: xshift: {W*self.sda.sal.xshift.item()/2}, yshift: {H*self.sda.sal.yshift.item()/2}, theta: {self.sda.sal.rot_theta.item()} ')

            # # Apply reflective boundary
            # with torch.no_grad():
            #     self.sda.sal.xshift.data = apply_reflective_boundary(self.sda.sal.xshift.data, 2*2/W)
            #     self.sda.sal.yshift.data = apply_reflective_boundary(self.sda.sal.yshift.data, 2*2/H)
            #     self.sda.sal.rot_theta.data = apply_reflective_boundary(self.sda.sal.rot_theta.data, 30*np.pi/180)

            # Collect outputs and loss
            output_list.append(outputs)
            losses.append(loss.item())
            xshifts.append(self.sda.sal.xshift.item())
            yshifts.append(self.sda.sal.yshift.item())
            angles.append(self.sda.sal.rot_theta.item())

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(np.array(losses)/self.base_loss)
            # axs[0].hlines(y=base_loss, xmin=0, xmax=len(np.array(losses).ravel()), linestyles='dashed')
            axs[0].set_title('Training Loss')
            axs[1].plot(W*(np.array(xshifts).ravel())/2)
            axs[1].plot(H*(np.array(yshifts).ravel())/2)
            axs[1].plot(angles)
            axs[1].hlines(y=[-self.params['Tx'], -self.params['Ty'], -self.params['theta']], xmin=0, xmax=len(np.array(xshifts).ravel()), linestyles='dashed', label='ground truth')
            axs[1].legend(['xshift-pred','yshift-pred', 'theta'])
            axs[1].set_title('Parameter Dynamics')
            axs[1].set_ylim([-2.0, 2.0])
            plt.savefig('learning.jpg')

        sources = outputs.detach().cpu()
        return sources, losses

    def search_fit_sda(self, emg_grid_transform, npoints=50, nepochs=50, lr=1e-4, device='cpu', loss='kurtosis', plot=1):
        ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''
        params = {'nepochs': nepochs, 'lr': lr}
        self.params.update(params)# keep track of chosing experimental parameter
        
        N, C, H, W = emg_grid_transform.shape
        self.sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.sda.parameters()),
                                                    lr=lr)                 

        # Collect output tensors
        output_list = []
        losses = []
        xshifts,yshifts,angles = [], [], []

        # Freeze all parameters except for SAL parameters
        for param in self.sda.parameters():
            param.requires_grad = False
        
        # Searching through initial conditions
        print('SAMPLING AND EVALUATING INITIAL CONDITIONS...')
        boundaries = torch.tensor([3.0, 3.0, 20*np.pi/180]) # symmetric for each dimension about zero
        losses = torch.zeros(npoints)
        engine = scipy.stats.qmc.LatinHypercube(d=3)
        init_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float32)-1 # scale from [0,1] to [-1, 1]
        init_params[:,0], init_params[:,1], init_params[:, 2] = boundaries[0]*init_params[:,0]/W, boundaries[1]*init_params[:,1]/H, boundaries[2]*init_params[:, 2]
        init_params = init_params.to(device)
        with torch.no_grad():
            for npoint in tqdm(range(npoints)):
                # Set initial conditions
                self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = init_params[npoint, :]

                # Evaluate loss function at given condition
                outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

                # Compute ICA Loss and backprop    
                loss = ica_loss(outputs)
                losses[npoint] = loss.item()

            losses = losses / self.base_loss # normalize by baseline loss
            self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = init_params[losses.argmax(), :] # get best initialization
            print(f'TOP 5 LOSS VALUES SAMPLED: {torch.topk(losses, k=5)}')

        # Make SAL parameters learnable
        self.sda.sal.xshift.requires_grad = True
        self.sda.sal.yshift.requires_grad = True
        self.sda.sal.rot_theta.requires_grad = True

        # Loop through the DataLoader
        print('TRAINING FROM BEST INIT. CONDITION...')
        losses = []
        for ne in tqdm(range(nepochs)):
            # Forward pass through the model
            outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

            # Compute ICA Loss and backprop    
            loss = ica_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            print('LOSS:', loss.item()/self.base_loss)
            optimizer.step()
            print(f'PARAMS: xshift: {W*self.sda.sal.xshift.item()/2}, yshift: {H*self.sda.sal.yshift.item()/2}, theta: {self.sda.sal.rot_theta.item()} ')

            # Collect outputs and loss
            output_list.append(outputs)
            losses.append(loss.item())
            xshifts.append(self.sda.sal.xshift.item())
            yshifts.append(self.sda.sal.yshift.item())
            angles.append(self.sda.sal.rot_theta.item())

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(losses)
            # axs[0].hlines(y=base_loss, xmin=0, xmax=len(np.array(losses).ravel()), linestyles='dashed')
            axs[0].set_title('Training Loss')
            axs[1].plot(W*(np.array(xshifts).ravel())/2)
            axs[1].plot(H*(np.array(yshifts).ravel())/2)
            axs[1].plot(angles)
            axs[1].hlines(y=[-self.params['Tx'], -self.params['Ty'], -self.params['theta']], xmin=0, xmax=len(np.array(xshifts).ravel()), linestyles='dashed', label='ground truth')
            axs[1].legend(['xshift-pred','yshift-pred', 'theta'])
            axs[1].set_title('Parameter Dynamics')
            axs[1].set_ylim([-2.0, 2.0])
            plt.savefig('learning.jpg')

        sources = outputs.detach().cpu()
        return sources, losses
    
    def bo_sda(self, emg_grid_transform, n_init_trials=10, n_updates=30, device='cpu', loss='kurtosis'):
        '''Bayesian optimization of the affine parameters given the loss function and SAL.'''
        # Define parameter bounds
        N, C, H, W = emg_grid_transform.shape
        self.sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        for param in self.sda.parameters():
            param.requires_grad = False
        
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        # Optimize the parameters using BOTorch
        # bounds = torch.tensor([[-2*3.0/W, -2*3.0/H, -20*np.pi/180], [2*3.0/W, 2*3.0/H, 20*np.pi/180]])  # symmetric bounds for parameters
        bounds = torch.tensor([[0.0,0.0,0.0], [1.0, 1.0, 1.0]])
        train_x = []  # Sampled parameter sets
        train_y = []  # Corresponding loss values

        # Initial random sampling
        print('SAMPLING INITIAL TRIALS FOR WARM START...')
        for _ in range(n_init_trials):
            params = torch.rand(3) * (bounds[1] - bounds[0]) + bounds[0]
            xshift, yshift, rot_theta = params.to(device)
            self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = 3*((2*xshift-1)*2)/W, 3*((2*yshift-1)*2)/H, (2*rot_theta-1)*20*np.pi/180 
            outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)
            loss = ica_loss(outputs).detach()/self.base_loss
            train_x.append(params)
            train_y.append(loss)

        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y).unsqueeze(-1)

        # UPDATING GP FOR BO
        for update in range(n_updates):
            # Step 2: Train GP with current data
            # kernel = RBFKernel(
            #     lengthscale_prior=LogNormalPrior(loc=-1.0, scale=0.5)  # Example prior
            # )
            # gp = SingleTaskGP(train_x, train_y, covar_module=kernel)
            gp = SingleTaskGP(train_x, train_y)
            gp.likelihood.noise = torch.tensor([1e-5], requires_grad=True) # lower noise so that we trust loss onservations more
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Step 3: Optimize acquisition function to find new candidate
            ei = LogExpectedImprovement(gp, best_f=train_y.max())
            new_x, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=50,
            )

            # Step 4: Evaluate the new candidate
            xshift, yshift, rot_theta = new_x.squeeze(0).to(device)
            self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = 3*((2*xshift-1)*2)/W, 3*((2*yshift-1)*2)/H, (2*rot_theta-1)*20*np.pi/180 
            outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)
            new_y = ica_loss(outputs).detach().cpu().view((1,1)) / self.base_loss

            # Step 5: Update training data
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])

            print(f"Update {update + 1}/{n_updates}: Loss = {new_y.item()}, Params = {new_x}")
            # Print best
            xbest, ybest, thetabest = train_x[train_y.argmax()]
            xbest, ybest, thetabest = 3*((2*xbest-1)*2)/W, 3*((2*ybest-1)*2)/H, (2*thetabest-1)*20*np.pi/180 
            print(f"Best: loss -> {train_y.max()} params -> {xbest}, {ybest}, {thetabest}")
            print()


        # Return the best parameters found
        best_index = train_y.argmax()
        return train_x[best_index], train_y.max()


    def loss_sampling(self, emg_grid_transform, num_points=20, loss='kurtosis', device='cpu'):
        ''' Method used to sample the loss landscape.'''
        N, C, H, W = emg_grid_transform.shape
        Tx, Ty = self.params['Tx'], self.params['Ty']
        self.params['loss'] = loss
        # emg_grid.requires_grad = True
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, ycrop=self.params['ycrop'], xcrop=self.params['xcrop'], extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()

        # Getting torch meshgrid
        # x = torch.linspace(-self.params['xcrop'], self.params['xcrop'], num_points) #.round(decimals=3)
        x = torch.linspace(-torch.tensor(2.0), torch.tensor(2.0), num_points)
        # x = torch.linspace(-1, 1, num_points) # theta
        # y = torch.linspace(-self.params['ycrop'], self.params['ycrop'], num_points) #.round(decimals=3)
        y = torch.linspace(-torch.tensor(2.0), torch.tensor(2.0), num_points)
        loss_arr = torch.zeros(y.shape[0], x.shape[0])

        # Sample parameters
        with torch.no_grad():
            for xidx, xi in enumerate(tqdm(x)):
                for yidx, yi in enumerate(y):
                    emg_grid_test = emg_grid_transform.clone()
                    sda.sal.yshift.copy_(torch.tensor(2*yi/H).to(device))
                    sda.sal.xshift.copy_(torch.tensor(2*xi/W).to(device))
                    # sda.sal.rot_theta.copy_(torch.tensor(xi).to(device))

                    outputs = sda(emg_grid_test.to(device))
                    loss = ica_loss(outputs)
                    loss_arr[yidx, xidx] = loss.item()
        
        plt.figure()
        plt.title(f'Kurtosis Loss Landscape (Post (y={Ty}, x={Tx}) translation)')
        ax = sns.heatmap(np.array(loss_arr)/self.base_loss, xticklabels=np.around((x).tolist(), 3), yticklabels=np.around((y).tolist(), 3))
        ax.set(xlabel='Circumferential Shifts (m)', ylabel='Longitudinal Shifts (m)')
        ax.text(np.where(np.array(x)>=-Tx)[0][0] + 0.5, np.where(y>=-Ty)[0][0]+0.5, 'X', color='green', ha='center', va='center', fontsize=16)
        plt.savefig('loss_landscape.jpg')
        print()

        return loss
    
    def theta_loss_sampling(self, emg_grid_transform, num_points=20, theta_max=30*np.pi/180, loss='kurtosis', device='cpu'):
        ''' Method used to sample the loss landscape.'''
        N, C, H, W = emg_grid_transform.shape
        true_theta = self.params['theta']
        self.params['loss'] = loss
        # emg_grid.requires_grad = True
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), whiten_mat=self.whiten_mat, sep_mat=self.sep_mat, ycrop=self.params['ycrop'], xcrop=self.params['xcrop'], extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()

        # Getting torch meshgrid
        thetas = torch.linspace(-theta_max, theta_max, num_points) #.round(decimals=3)
        loss_arr = torch.zeros(thetas.shape)

        # Sample parameters
        with torch.no_grad():
            for tidx, theta in enumerate(tqdm(thetas)):
                emg_grid_test = emg_grid_transform.clone()
                sda.sal.rot_theta.copy_(torch.tensor(theta).to(device))
                outputs = sda(emg_grid_test.to(device))
                loss = ica_loss(outputs)
                loss_arr[tidx] = loss.item()
        
        plt.figure()
        plt.title(f'Kurtosis Loss Landscape (Post (theta={true_theta}) rotation)')
        plt.plot(thetas, loss_arr.detach().numpy())
        plt.vlines(x=-self.params['theta'], ymin=torch.min(loss_arr).item(), ymax=0, linestyles='dashed', label='ground truth')
        plt.savefig('theta_loss_landscape.jpg')
        print()


if __name__ == '__main__':

    # Experimental parameters
    mu_count=20
    H, W, L =25, 10, 50
    R = 16
    fxmax=0.9 # normalized spatial cutoff frequency
    sampfactor=15

    duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
    Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
    SNR = 1 # SNR for synthetic EMG

    Tx, Ty, theta, xscale, yscale = -1.5, 1.5, -15*np.pi/180, 1, 1 # affine parameters applied
    # Training params
    nepochs=50
    lr = 5e-3
    # batch_size = 2048
    loss = 'kurtosis'
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    exp = SDAExperiment()

    print('GENERATING MUAPS....')
    muaps = exp.generate_gaussian_muaps(mu_count, H, W, L, fxmax, sampfactor) # generate MUAPs

    print('GENERATING SPIKE TRAINS...')
    spts, dts = exp.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains

    print('GENERATE EMG...')
    emg = exp.generate_emg(spts, muaps) # make synthetic EMG from simulated MUAPs and spike trains
    emg = exp.add_noise(emg, SNR) # add noise to synthetic signal
    emg_grid = exp.make_grid(emg) # reshape into EMG grid

    print('APPLY TRANSFORM...')
    emg_grid_transform = exp.apply_affine(emg_grid, Tx, Ty, theta, xscale, yscale, sampfactor)

    print('DOWNSAMPLING...')
    emg_grid, emg_grid_transform = exp.downsample_grid(emg_grid, sampfactor), exp.downsample_grid(emg_grid_transform, sampfactor)
    muaps = exp.downsample_muaps(muaps, sampfactor)

    print('GET SEPARATION VECTORS...')
    # torch.set_default_dtype(torch.float64) # Set pytorch default to float64
    B = exp.get_separation_vectors(muaps, R=R)
    print('GET WHITENING MATRIX AND TRANSFORMING SEPARATION VECTORS...')
    # emg_grid = exp.grid_crop(emg_grid, xcrop=0, ycrop=0) # crop grid so that we get appropriate shapes for the EMG
    source_est = exp.get_whiten_matrix(emg_grid, B) # stores separation vector and whiten matrices as attributes
    print()

    # print('TRAINING SDA MODULE...')

    # exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device)
    # sources, losses = exp.fit_sda(emg_grid_transform.to(torch.float32), nepochs, lr, device=device, loss=loss)

    # num_points=100
    # print(f'THETA SAMPLING LOSS LANDSCAPE ({num_points} samples)...')
    # losses = exp.theta_loss_sampling(emg_grid_transform.to(torch.float32), num_points=num_points, device=device)

    # num_points=20
    # print(f'SAMPLING LOSS LANDSCAPE ({num_points}x{num_points})...')
    # # emg_grid_transform = exp.apply_affine(emg_grid_transform, 0, 0, -theta, 1, 1, sampfactor=1) #REVERT ROTATION AND LOOK AT LOSS LANDSCAPE
    # exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device)
    # losses = exp.loss_sampling(emg_grid_transform.to(torch.float32), num_points=num_points, device=device)

    # SDA BO approach
    # exp.get_base_loss(emg_grid.to(torch.float64), loss=loss, device=device)
    # params, loss = exp.bo_sda(emg_grid_transform.to(torch.float64), n_init_trials=1, n_updates=20, device='cuda')

    print('TRAINING SDA MODULE...')
    exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device)
    sources, losses = exp.search_fit_sda(emg_grid_transform.to(torch.float32), npoints=30, nepochs=nepochs, lr=lr, device=device, loss=loss)
    print()
