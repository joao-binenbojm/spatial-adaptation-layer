import numpy as np
from math import ceil
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import seaborn as sns
from sklearn.cluster import KMeans
from torch_pso import ParticleSwarmOptimizer

from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg, get_silohuette, maxk
# from sal_decomposition.MUEdit.processing_tools import batch_process_filters as get_pulse_trains
from loss_functions import KurtosisLoss, NegentropyLoss
from sal_decomposition.sda import SpatialDecompositionAdaptation
from networks_utils import SpatialAdaptation

class SDAExperiment:

    def __init__(self):
        self.params = {}

    def generate_gaussian_muaps(self, mu_count, H, W, L, fxmax, sampfactor=10):
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
        params = {'mu_count': mu_count, 'duration': duration, 'Tmean':Tmean, 'ISV':ISV}
        self.params.update(params)# keep track of chosing experimental parameters
        
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

    def generate_emg(self, spts, muaps, R, device='cpu'):
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
    
    def apply_affine(self, emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, sampfactor=10):
        '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''
        params = {'Tx': Tx, 'Ty': Ty, 'theta': theta, 'xscale':xscale, 'yscale':yscale, 'sampfactor': sampfactor}
        self.params.update(params)# keep track of chosing experimental parameters

        N, C, H, W = emg_grid.shape
        Tx, Ty = torch.tensor(2*Tx*sampfactor/W), torch.tensor(2*sampfactor*Ty/H) # Normalize translation values automatically
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

    def downsample_grid(self, emg_grid, sampfactor=10):
        '''Downsample EMG signal given the originally used sampfactor.'''
        return emg_grid[:, :, ::sampfactor, ::sampfactor] # downsamples EMG grid spatially

    def downsample_muaps(self, muaps, sampfactor=10):
        '''Downsample the MUAPs before generating separation vectors.'''
        return muaps[:, ::sampfactor, ::sampfactor, :] # downsample muaps along spatial coordinates

    def get_separation_vectors(self, muaps, R=None, xcrop=0, ycrop=0):
        ''' Based on MUAPs, just generate the separation vectors neccessary.'''
        R = R if R is not None else muaps.shape[-1]
        delay = torch.ceil(torch.tensor(((self.params['L'] + R) / 2))).to(torch.int) - 1 # delay introduced by causality of triggering process
        params = {'R': R, 'xcrop': xcrop, 'ycrop': ycrop, 'delay': delay}
        self.params.update(params)# keep track of chosing experimental parameters

        N, H, W, L = muaps.shape
        if R is None: R = L
        Nch = (H-2*ycrop)*(W-2*xcrop)
        B = torch.zeros(N, Nch*R)
        for mdx in range(N):
            for l in range(R):
                B[mdx, l*Nch:(l+1)*Nch] = muaps[mdx, ycrop:H-ycrop, xcrop:W-xcrop, delay - l].ravel() # MUAP reversed is the separation vector itself!
            B[mdx, :] = B[mdx, :] / (torch.linalg.vector_norm(B[mdx, :]) + 1e-12) # make a unit vector
        return B
    
    # def get_separation_vectors_avg(self, whitened_emg, dts, R=None, xcrop=0, ycrop=0):
    #     ''' Based on MUAPs, just generate the separation vectors neccessary.'''
    #     R = R if R is not None else muaps.shape[-1]
    #     B = torch.zeros(len(dts), whitened_emg.shape[0]*R)
    #     # N, H, W, L = muaps.shape
    #     if R is None: R = L
    #     # Nch = (H-2*ycrop)*(W-2*xcrop)
    #     self.delay = torch.ceil(torch.tensor(((L + R) / 2))).to(torch.int) # center extended delays about center of MUAP
    #     B = torch.zeros(len(dts), whitened_emg.shape[0])
    #     for mdx in range(len(dts)):
    #         mu_dts = dts[mdx][dts[mdx] <= (dts[mdx].max() - self.delay)]
    #         B[mdx, :] = whitened_emg[:, mu_dts + self.delay].mean(dim=1) # spike triggered averaging
    #         B[mdx, :] = B[mdx, :] / (torch.linalg.vector_norm(B[mdx, :]) + 1e-12)
    #     return B

    def get_inv_cov(self, signal, explained_var=0.99):
    
        """ Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization. """
        cov_mat = np.cov(np.squeeze(signal),bias=True)
        print('FINISHED GETTING COVARIANCE MATRIX...')
        # get the eigenvalues and eigenvectors of the covariance matrix
        evalues, evectors  = scipy.linalg.eigh(cov_mat)
        print('FINISHED GETTING EIGENDECOMPOSITION...')
        # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)

        # penalty = np.mean(sorted_evalues[len(sorted_evalues)//2:]) # int won't wokr for odd numbers
        # penalty = max(0, penalty)

        # rank_limit = np.sum(evalues > penalty)-1
        # if rank_limit < np.shape(signal)[0]:

        #     hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2
        # # use the rank limit to segment the eigenvalues and the eigenvectors
        # evectors = evectors[:,evalues > hard_limit]
        # evalues = evalues[evalues>hard_limit]
        # sorted_evalues = np.sort(evalues)[::-1]
        sorted_idxs = np.argsort(evalues)[::-1] # sort in descending order
        evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
        cum_explained_var = evalues.cumsum() / evalues.sum()
        evalues, evectors = evalues[cum_explained_var <= explained_var], evectors[:, cum_explained_var <= explained_var]

        inv_cov = evectors @ np.diag(1 / (evalues)) @ np.transpose(evectors)
        return inv_cov
    
    def process_sep_mat(self, emg_grid, B, R):
        ''' Get whiten matrix based on EMG and apply transpose to separation vectors.'''
        self.params.update({'R': R})
        # Get whitened extended observations
        emg = np.array(emg_grid).squeeze().reshape(emg_grid.shape[0], -1).T
        extended_emg_template = np.zeros((self.params['R']*emg.shape[0], emg.shape[1] + self.params['R'] - 1))
        extended_emg = extend_emg(extended_emg_template, emg, self.params['R'])
        inv_cov = self.get_inv_cov(extended_emg)

        # Get separation matrix based on whitened observations
        print('GETTING SEPARATION VECTORS...')
        B = B @ inv_cov
        self.sep_mat = torch.tensor(B).to(torch.float32)
        sources = self.sep_mat @ torch.tensor(extended_emg).to(torch.float32)
        return sources
    
    def get_source_estimate(self, emg_grid):
        '''Estimates sources based on constructed separation vector.'''
        emg = np.array(emg_grid).squeeze().reshape(emg_grid.shape[0], -1).T
        extended_emg_template = np.zeros((self.params['R']*emg.shape[0], emg.shape[1] + self.params['R'] - 1))
        extended_emg = extend_emg(extended_emg_template, emg, self.params['R'])
        sources = self.sep_mat @ torch.tensor(extended_emg).to(torch.float32)
        return sources

    def get_base_loss(self, emg_grid, loss='kurtosis', device='cpu'):
        '''Getting base loss.'''
        N, C, H, W = emg_grid.shape
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        with torch.no_grad():
            self.base_loss = ica_loss(sda(emg_grid.to(device))).item()

    def search_fit_sda(self, emg_grid_transform, npoints=50, nepochs=50, lr=1e-4, device='cpu', loss='kurtosis', frozen_sep_mat=False, plot=1):
        ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''
        params = {'nepochs': nepochs, 'lr': lr}
        self.params.update(params)# keep track of chosing experimental parameter
        
        N, C, H, W = emg_grid_transform.shape
        self.sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.sda.parameters()),
                                                    lr=lr)                 

        # Collect output tensors
        output_list = []
        losses = []
        xshifts,yshifts,angles,xscales,yscales = [], [], [], [], []

        # Freeze all parameters except for SAL parameters
        for param in self.sda.parameters():
            param.requires_grad = False
        
        # Searching through initial conditions
        print('SAMPLING AND EVALUATING INITIAL CONDITIONS...')
        boundaries = torch.tensor([3.0, 3.0, 20*np.pi/180, 0.2, 0.2]) # symmetric for each dimension about zero
        losses = torch.zeros(npoints)
        engine = scipy.stats.qmc.LatinHypercube(d=5)
        init_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float32)-1 # scale from [0,1] to [-1, 1]
        init_params[:,0], init_params[:,1], init_params[:, 2] = 2*boundaries[0]*init_params[:,0]/W, 2*boundaries[1]*init_params[:,1]/H, boundaries[2]*init_params[:, 2]/np.pi
        init_params[:, 3] = torch.pow((1 + torch.abs(init_params[:, 3])*boundaries[3]), torch.sign(init_params[:, 3]) ) # generates scalings appropriately
        init_params[:, 4] = torch.pow((1 + torch.abs(init_params[:, 4])*boundaries[4]), torch.sign(init_params[:, 4]) )

        init_params = init_params.to(device)
        self.sda.train() # leave batch norm parameters adaptive
        with torch.no_grad():
            for npoint in tqdm(range(npoints)):
                # Set initial conditions
                self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = init_params[npoint, :3]
                self.sda.sal.xscale.data, self.sda.sal.yscale.data = init_params[npoint, 3:]
                # Evaluate loss function at given condition
                outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

                # Compute ICA Loss and backprop    
                loss = ica_loss(outputs)
                losses[npoint] = loss.item()

            if npoints > 0:
                losses = losses / self.base_loss # normalize by baseline loss
                self.sda.sal.xshift.data, self.sda.sal.yshift.data, self.sda.sal.rot_theta.data = init_params[losses.argmax(), :3] # get best initialization
                self.sda.sal.xscale.data, self.sda.sal.yscale.data = init_params[losses.argmax(), 3:]
                print(f'TOP 5 LOSS VALUES SAMPLED: {torch.topk(losses, k=torch.min(torch.tensor([npoints, 5])))}')

        # Make SAL parameters learnable
        # for param in self.sda.sal.parameters():
        if frozen_sep_mat:
            for param in self.sda.sal.parameters():
                param.requires_grad = True        
        else:
            for param in self.sda.parameters():
                param.requires_grad = True

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
            print(f'PARAMS:\n xshift: {W*self.sda.sal.xshift.item()/2}, yshift: {H*self.sda.sal.yshift.item()/2}, theta: {self.sda.sal.rot_theta.item()} ')
            print(f'xscale: {self.sda.sal.xscale.item()}, yscale: {self.sda.sal.yscale.item()}')
            # Collect outputs and loss
            output_list.append(outputs)
            losses.append(loss.item())
            xshifts.append(self.sda.sal.xshift.item())
            yshifts.append(self.sda.sal.yshift.item())
            angles.append(self.sda.sal.rot_theta.item())
            xscales.append(self.sda.sal.xscale.item())
            yscales.append(self.sda.sal.yscale.item())

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(losses)
            axs[0].set_title('Training Loss')
            axs[1].plot(W*(np.array(xshifts).ravel())/2)
            axs[1].plot(H*(np.array(yshifts).ravel())/2)
            axs[1].plot(angles)
            axs[1].plot(xscales)
            axs[1].plot(yscales)
            axs[1].hlines(y=[-self.params['Tx'], -self.params['Ty'], -self.params['theta'], 1/self.params['xscale'], 1/self.params['yscale']],
                          xmin=0, xmax=len(np.array(xshifts).ravel()), linestyles='dashed', label='ground truth')
            axs[1].legend(['xshift-pred','yshift-pred', 'theta', 'xscale', 'yscale'])
            axs[1].set_title('Parameter Dynamics')
            axs[1].set_ylim([-3.0, 3.0])
            plt.savefig('learning.jpg')

        # Get final outputs, i.e. optimal souces
        with torch.no_grad():
            final_outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

        sources = final_outputs.detach().cpu()
        return sources, losses

    def particle_swarm_sda(self, emg_grid_transform, nepochs=100, n_particles=5, w=0.8, c1=0.1, c2=0.1, max_param=1.0, min_param=-1.0, device='cpu', loss='kurtosis', plot=1):
        ''' Fit SDA to emg_grid data to find optimal affine parameters. If plot, plot learning of all parameters and loss over iterations.'''
        params = {'nepochs': nepochs, 'lr': lr}
        self.params.update(params)# keep track of chosing experimental parameter
        
        N, C, H, W = emg_grid_transform.shape
        self.sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=self.sep_mat, extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()
        optimizer = ParticleSwarmOptimizer(self.sda.sal.parameters(), inertial_weight=w,
                                            num_particles=n_particles, cognitive_coefficient=c1,
                                            social_coefficient=c2, max_param_value=max_param, min_param_value=min_param)             
        # Collect output tensors
        output_list = []
        losses = []
        xshifts,yshifts,angles,xscales,yscales = [], [], [], [], []

        for param in self.sda.sal.parameters():
            param.requires_grad = True        

        # Loop through the DataLoader
        print('PSO optimization')
        losses = []
        with torch.no_grad():
            for ne in tqdm(range(nepochs)):
                # Forward pass through the model
                outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

                # Compute ICA Loss and backprop
                def closure():
                    optimizer.zero_grad()
                    return ica_loss(outputs)
                
                optimizer.step(closure)
                # loss = ica_loss(outputs)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # Display updates
                print('LOSS:', loss.item()/self.base_loss)
                print(f'PARAMS:\n xshift: {W*self.sda.sal.xshift.item()/2}, yshift: {H*self.sda.sal.yshift.item()/2}, theta: {self.sda.sal.rot_theta.item()} ')
                print(f'xscale: {self.sda.sal.xscale.item()}, yscale: {self.sda.sal.yscale.item()}')
                # Collect outputs and loss
                output_list.append(outputs)
                losses.append(loss.item())
                xshifts.append(self.sda.sal.xshift.item())
                yshifts.append(self.sda.sal.yshift.item())
                angles.append(self.sda.sal.rot_theta.item())
                xscales.append(self.sda.sal.xscale.item())
                yscales.append(self.sda.sal.yscale.item())

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(losses)
            axs[0].set_title('Training Loss')
            axs[1].plot(W*(np.array(xshifts).ravel())/2)
            axs[1].plot(H*(np.array(yshifts).ravel())/2)
            axs[1].plot(angles)
            axs[1].plot(xscales)
            axs[1].plot(yscales)
            axs[1].hlines(y=[-self.params['Tx'], -self.params['Ty'], -self.params['theta'], 1/self.params['xscale'], 1/self.params['yscale']],
                          xmin=0, xmax=len(np.array(xshifts).ravel()), linestyles='dashed', label='ground truth')
            axs[1].legend(['xshift-pred','yshift-pred', 'theta', 'xscale', 'yscale'])
            axs[1].set_title('Parameter Dynamics')
            axs[1].set_ylim([-3.0, 3.0])
            plt.savefig('learning.jpg')

        # Get final outputs, i.e. optimal souces
        with torch.no_grad():
            final_outputs = self.sda(emg_grid_transform.to(device)).to(device)  # Shape will be (batch_size, num_classes)

        sources = final_outputs.detach().cpu()
        return sources, losses
    
    def loss_sampling(self, emg_grid_transform, num_points=20, loss='kurtosis', device='cpu'):
        ''' Method used to sample the loss landscape.'''
        N, C, H, W = emg_grid_transform.shape
        Tx, Ty = self.params['Tx'], self.params['Ty']
        self.params['loss'] = loss
        # emg_grid.requires_grad = True
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=self.sep_mat, ycrop=self.params['ycrop'], xcrop=self.params['xcrop'], extension_factor=self.params['R']).to(device)
        if loss == 'kurtosis':
            ica_loss = KurtosisLoss()
        else:
            ica_loss = NegentropyLoss()

        # Getting torch meshgrid
        x = torch.linspace(-torch.tensor(3.0), torch.tensor(3.0), num_points)
        y = torch.linspace(-torch.tensor(3.0), torch.tensor(3.0), num_points)
        loss_arr = torch.zeros(y.shape[0], x.shape[0])

        # Sample parameters
        sda.train() # leave layer norm adaptive and running
        with torch.no_grad():
            for xidx, xi in enumerate(tqdm(x)):
                for yidx, yi in enumerate(y):
                    emg_grid_test = emg_grid_transform.clone().detach()
                    sda.sal.yshift.copy_(torch.tensor(2*yi/H).to(device))
                    sda.sal.xshift.copy_(torch.tensor(2*xi/W).to(device))

                    outputs = sda(emg_grid_test.to(device))
                    loss = ica_loss(outputs)
                    loss_arr[yidx, xidx] = loss.item()
        
        plt.figure()
        # plt.title(f'Kurtosis Loss Landscape (Post (y={Ty}, x={Tx}) translation)')
        # ax = sns.heatmap(np.array(loss_arr)/self.base_loss, xticklabels=np.around((x).tolist(), 3), yticklabels=np.around((y).tolist(), 3))
        ax = sns.heatmap(np.array(loss_arr)/self.base_loss)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # ax.set(xlabel='Circumferential Shifts (mm)', ylabel='Longitudinal Shifts (mm)')
        ax.text(np.where(np.array(x)>=-Tx)[0][0] + 0.5, np.where(y>=-Ty)[0][0]+0.5, 'X', color='green', ha='center', va='center', fontsize=16)
        
        plt.savefig('loss_landscape.jpg')
        print()

        return loss_arr
    
    def theta_loss_sampling(self, emg_grid_transform, num_points=20, theta_max=30*np.pi/180, loss='kurtosis', device='cpu'):
        ''' Method used to sample the loss landscape.'''
        N, C, H, W = emg_grid_transform.shape
        true_theta = self.params['theta']
        self.params['loss'] = loss
        # emg_grid.requires_grad = True
        sda = SpatialDecompositionAdaptation(grid_shape=(H, W), sep_mat=self.sep_mat, ycrop=self.params['ycrop'], xcrop=self.params['xcrop'], extension_factor=self.params['R']).to(device)
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

    def get_silohuette(self, sources_pred, distance=4):
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

    def spike_scores(self, dts, dts_pred):
        ''' For each motor unit, compute the spiking accuracy, sensitivity and precision.'''
        scores = {'sensitivity': np.zeros(len(dts)), 'precision': np.zeros(len(dts))}
        for mu_idx in range(len(dts)):
            gt, pred = set(dts[mu_idx].tolist()), set(dts_pred[mu_idx] - self.params['delay'].item()) # account for delay induced
            tps = len(gt.intersection(pred)) # intersection of discharge times is true positives
            fps = len(pred.difference(gt)) # false positives = dts in pred not in gt
            fns = len(gt.difference(pred)) # false negatives = dts in gt not in pred
            scores['sensitivity'][mu_idx] = tps / (tps + fns) # how real spikes are missed
            scores['precision'][mu_idx] = tps / (tps + fps) # how many fake spikes are assumed
        return scores

if __name__ == '__main__':

    # Experimental parameters
    mu_count=20
    H, W, L = 25, 10, 50
    R = 16
    fxmax=125 / 125 # normalized spatial cutoff frequency
    sampfactor=15

    duration = 20000 # number of time samples in EMG, equivalent of 10s with fs=2000Hz
    Tmean, ISV = 60, 0.2 # sample statistics of spikes # equivalent of 30Hz with fs=2000Hz
    SNR = 1 # SNR for synthetic EMG

    Tx, Ty, theta, xscale, yscale = -1.5, 2.5, 15*np.pi/180, 1.15, 0.85 # affine parameters applied
    # Tx, Ty, theta, xscale, yscale = -1.5, 2.5, 0, 1, 1 # affine parameters applied
    # Tx, Ty, theta, xscale, yscale = -1.5, 2.2, 8*np.pi/180, 1.0, 1.0
    # Training params
    nepochs=120
    lr = 5e-3
    loss = 'kurtosis'
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    with torch.no_grad():
        exp = SDAExperiment()

        print('GENERATING MUAPS....')
        muaps = exp.generate_gaussian_muaps(mu_count, H, W, L, fxmax, sampfactor) # generate MUAPs

        print('GENERATING SPIKE TRAINS...')
        spts, dts = exp.generate_spike_trains(mu_count, duration, Tmean, ISV) # Generate spike trains

        print('GENERATE EMG...')
        emg = exp.generate_emg(spts, muaps, R=R) # make synthetic EMG from simulated MUAPs and spike trains
        emg = exp.add_noise(emg, SNR) # add noise to synthetic signal
        # emg = (emg - emg.mean(dim=2, keepdim=True)) #/ (emg.std(dim=2, keepdim=True) - 1e-9)
        emg_grid = exp.make_grid(emg) # reshape into EMG grid

        print('APPLY TRANSFORM...')
        emg_grid_transform = exp.apply_affine(emg_grid.detach().clone(), Tx, Ty, theta, xscale, yscale, sampfactor)

        print('DOWNSAMPLING...')
        emg_grid, emg_grid_transform = exp.downsample_grid(emg_grid, sampfactor), exp.downsample_grid(emg_grid_transform, sampfactor)
        muaps = exp.downsample_muaps(muaps, sampfactor)

        print('CENTERING...')
        mean = (emg_grid.mean(dim=0, keepdim=True) + emg_grid_transform.mean(dim=0, keepdim=True)) / 2
        emg_grid, emg_grid_transform = emg_grid - mean, emg_grid_transform - mean

        # print('STANDARDIZING CHANNELS...')
        # emg_grid = (emg_grid - emg_grid.mean(dim=0, keepdim=True)) / (emg_grid.std(dim=0, keepdim=True) + 1e-12)
        # emg_grid_transform = (emg_grid_transform - emg_grid_transform.mean(dim=0, keepdim=True)) / (emg_grid_transform.std(dim=0, keepdim=True) + 1e-12)

        print('GET SEPARATION VECTORS...')
        B = exp.get_separation_vectors(muaps, R=R)
        source_est = exp.process_sep_mat(emg_grid, B, R=R)
        print()

        # Get score estimates just after the transformation
        source_est_transform = exp.get_source_estimate(emg_grid_transform)
        pred_dts, sils = exp.get_silohuette(source_est_transform.detach().cpu().numpy().T)
        scores = exp.spike_scores(dts, pred_dts)
        print('SILS:', sils)
        print()
        print('SCORES:', scores)
        print('AVGs:', np.mean(scores['sensitivity']), np.mean(scores['precision']))

    # # Loss sampling
    # with torch.no_grad():
    #     num_points=20
    #     print(f'SAMPLING LOSS LANDSCAPE ({num_points}x{num_points})...')
    #     exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device)
    #     losses = exp.loss_sampling(emg_grid_transform.to(torch.float32), num_points=num_points, loss=loss, device=device)

    with torch.no_grad():
        print('TRAINING SDA MODULE...')
        exp.get_base_loss(emg_grid.to(torch.float32), loss=loss, device=device)
    sources, losses = exp.search_fit_sda(emg_grid_transform.to(torch.float32), npoints=nepochs, nepochs=nepochs//2, lr=lr, device=device, loss=loss, frozen_sep_mat=True)
    print()

    # Performance metrics based on output losses
    pred_dts, sils = exp.get_silohuette(sources.detach().cpu().numpy())
    scores = exp.spike_scores(dts, pred_dts)
    print('SILS:', sils)
    print()
    print('SCORES:', scores)
    print('AVGs:', np.mean(scores['sensitivity']), np.mean(scores['precision']))