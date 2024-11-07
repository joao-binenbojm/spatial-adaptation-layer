from scipy.io import loadmat
from scipy.spatial.distance import mahalanobis
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from time import time

def snake_grid(linear_vector, shape, invert=False):
    snaked = linear_vector.reshape((shape[1], shape[0]))
    snaked[1::2] = snaked[1::2, ::-1]
    snaked = snaked.T
    if not invert:
        return snaked
    else:
        return np.flipud(snaked)

def eigen_cutoff(explained_var, thrs=0.99):
    ''' Determines eigenvalue at which the cumulative sum of explained variance reaches a designated threshold.'''
    cum_var = np.cumsum(explained_var)
    loc = np.argwhere(cum_var >= thrs)[0,0] # first index where explained variance goes above threshold
    return loc + 1 # eigenvalue at which explained variance goes above threshold

def sortMUAPs(muaps: np.ndarray, J: int):
    ''' Given MUAPs data array, and extracts indices of MUAPs in ascending order of amplitude variance.'''
    print('SORTING MUAPs...')
    totvars = [] # variance of MUAP sizes across electrodes
    for jdx in range(len(muaps[0, :])): # for each motor unit
        totvar = []
        for cdx in range(len(muaps[0,0][0,:])):
            for rdx in range(muaps[0,0][0,0].shape[0]):
                totvar.append(np.var(muaps[0, jdx][0, cdx][rdx,:])) # variability of given MU/Channel time impulse response
        totvars.append(totvar)
    avg_totvars = np.mean(totvars, axis=1) # variability of MU averaged across channels
    MUranks = np.flip(np.argsort(avg_totvars)) # indices for sorting muaps based on MUAP size
    return MUranks

def getH(data: np.ndarray, K: int, J: int) -> np.ndarray:
    ''' Returns mixing matrix based on simulated MUAPS.'''
    ncols = len(data['MUAPs'][0, 0][0]) # number of columns in simulated grid
    nrows, L = data['MUAPs'][0, 0][0, 0].shape # number of rows in simulated grid and length of MUAP
    M = ncols*nrows # total number of channels in the grid
    H = np.zeros((K*M, J*(L + K - 1))) # initialize mixing matrix
    grid = snake_grid(np.arange(90), shape=(10,9))
    # MUranks = sortMUAPs(data['MUAPs'], J=J) # get indices of MUs sorted by average variability

    for jdx in range(J): # for each motor unit
        # jrank = MUranks[jdx] # which motor unit to extract
        for mdx in range(M): # for each channel
            for kdx in range(K): # for each delayed repetition
                coords = np.argwhere(grid == mdx)
                row, col = coords[0,0], coords[0, 1]
                muap = data['MUAPs'][0, jdx][0, col][row, :] # get the current muap
                interval = (L + K - 1)*jdx # interval between each AP shape
                H[mdx*K + kdx, interval+kdx:interval+kdx+L] = muap.ravel() # adding motor unit shapes to mixing matrix
    return H

def IA(EMG: np.ndarray, K:int =1) -> np.ndarray:
    ''' Computes the index of activity based on the EMG and chosen extension factor.'''

    # Unfold the simulation EMG shape
    if np.prod(EMG.shape) == 90:
        EMG = np.array([EMG[idx, jdx][0,:] for idx in range(10) for jdx in range(9)])
    T = EMG.shape[1] # number of samples of EMG signal

    # Computing the mahalanobis distance of each vector
    EMG = np.concatenate((np.zeros((EMG.shape[0], K-1)), EMG), axis=1) # zero-padding given K-1 delayed repetitions
    EMG_extend = np.zeros((EMG.shape[0]*K, EMG.shape[1])) # where to store extended observations

    # Extend observations and compute covariance matrix
    print('Extracting covariance matrix...')
    for t in range(T): # for each sample
        delreps = EMG[:, t : t+K] # get all the samples needed for extended observations
        y = np.flip(delreps, axis=1).ravel() # flip observations to match representation in original paper
        EMG_extend[:, t]  = y # store in extended observations matrix
    cov_y = np.cov(EMG_extend) # extended observations covariance matrix
    cov_y_inv = np.linalg.inv(cov_y) # inverse of extended observations
    EMG_extend = EMG_extend - EMG_extend.mean(axis=1).reshape(-1, 1) # subtract mean from each row

    # Extract the index of activity at all given timepoints
    print('Extracting IA...')
    gamma = np.zeros(EMG.shape[1]) # as many values as samples available of EMG
    for t in range(T):
        gamma[t] = EMG_extend[:, t].T @ cov_y_inv @ EMG_extend[:, t] 
    
    return gamma

def generate_MUAPs(M, J, L):
    ''' Generate sine waves with guaranteed different frequencies.'''
    MUAPs = np.zeros((M, J, L))
    intervals = np.linspace(start=int(L/3), stop=L, num=M*J).reshape((M, J))
    for mdx in range(M):
        for jdx in range(J):
            f = 1/intervals[mdx, jdx] # frequency of given sine waveform
            MUAP = np.sin(2*np.pi*f*np.arange(intervals[mdx, jdx]))
            start = (L - len(MUAP))//2
            MUAPs[mdx, jdx, start:(start + len(MUAP))] = MUAP # add MUAP shape
            # plt.figure()
            # plt.plot(MUAPs[mdx, jdx, :])
            # plt.show()
    return MUAPs

def generate_random_MUAPs(M, J, L):
    ''' Generate random gaussian waveforms to use as independent 'MUAPs'.'''
    MUAPs = np.zeros((M, J, L))
    for mdx in range(M):
        for jdx in range(J):
            MUAPs[mdx, jdx, :] = np.random.normal(size=L) # add MUAP shape
    return MUAPs

def get_mixing_matrix(MUAPs, K=1):
    '''Based on genered MUAPs, get H.'''
    M, J, L = MUAPs.shape
    H = np.zeros((K*M, J*(L + K - 1)))
    for mdx in range(M): # for each channel
        for kdx in range(K): # for each delayed repetition
            for jdx in range(J): # for each motor unit
                interval = (L + K - 1)*jdx # interval between each AP shape
                H[mdx*K + kdx, interval+kdx:interval+kdx+L] = MUAPs[mdx, jdx, :].ravel() # adding motor unit shapes to mixing matrix
    
    return H

# def generate_spikes(N, J, fr, fs):
#     '''Generates spikes randomly by creating spike trains with the chosen properties.'''
#     spike_trains = np.zeros((N, J))

#     Tsamp = int(fs/fr) # number of samples in interval between spikes
#     interval = int(fs/(fr*J)) # number of samples between firing rates of consecutive motor units

#     init_train = []
#     while len(init_train) < N: # generate spike train
#         if N - len(init_train) > Tsamp: 
#             init_train.extend([0]*(Tsamp - 1) + [1])
#         else:
#             init_train.extend([0]*(N - len(init_train)))

#     # Compute individual spike trains
#     spike_trains[:, 0] = init_train # initial spike train
#     for jdx in range(1, J):
#         spike_trains[:, jdx] = init_train[-interval*jdx:] + init_train[:-interval*jdx] # circularly roll signal

#     return spike_trains

def get_random_spikes(N, J, fr, fs):
    dt = 1/fs # in seconds
    nBins = N # number of samples
    nTrials = J # number of spikes trains
    spike_mat = (np.random.uniform(size=(nTrials, nBins)) < fr*dt)
    return spike_mat

def get_sig_out(H, spike_trains, L, K):
    ''' Based on H matrix and simulated spike train, generates synthetic EMG signal.'''
    T = spike_trains.shape[0] # number of samples we have
    spike_trains = np.concatenate((np.zeros((L+K-1, spike_trains.shape[1])), spike_trains), axis=0) # zero-padding given K-1 delayed repetitions
    spikes_ext = np.zeros((spike_trains.shape[1]*(L+K-1), T)) # where to store extended spike trains

    # Extend observations and compute covariance matrix
    print('Extracting extended spike train vector...')
    for t in range(T): # for each sample
        delreps = spike_trains[t : t + L + K - 1, :].T # get all the samples needed for extended observations
        spike_ext = np.flip(delreps, axis=1).ravel() # flip observations to match representation in original paper
        spikes_ext[:, t]  = spike_ext # store in extended observations matrix

    # Compute extended observation vector, and get only non-repeated observations
    Y_ext = H @ spikes_ext
    Y = Y_ext[np.arange(start=0, stop=H.shape[0], step=K), :]

    return Y

if __name__ == '__main__':
    DIR = '../SynthSigs'
    name = 'SynthMUAP_BB_lib1_F5-5-5_Len20_ramp1_SNR10_20-Nov-2023_16_50.mat'
    data = loadmat(os.path.join(DIR, name))
    print(data.keys())
    fs = data['fsamp'].ravel()
    firings = data['sFirings']
    sig_out = data['sig_out']
    frs = [(fs/(np.diff(firings[0, idx].ravel()).mean())) for idx in range(firings.shape[1])]

    # # Get index of activity
    # L, K = 256, 1
    # kernel = np.ones(L + K - 1)
    # ia = IA(sig_out, K=K)
    # plt.figure()
    # plt.plot(ia)
    # plt.show()

    # # Deconvolve with square wave
    # ia_deconv,_ = signal.deconvolve(ia, kernel)
    # plt.figure()
    # plt.plot(ia_deconv)
    # plt.show()

    # plt.figure()
    # plt.plot(frs)
    # plt.ylim([0, 15])
    # plt.show()

    # plt.figure()
    # plt.plot(sig_out[0,0].ravel())
    # plt.show()

    # Validating the global index of activity
    T, fs = 1, 2048
    L, K = 40,1 
    N = int(T * fs) # number of samples
    J = 1 # 20 MUs
    # spike = generate_spikes(N=N, J=J, fr=12, fs=fs)
    spike = get_random_spikes(N, J, fr=20, fs=fs).T
    spike = (spike - spike.mean(axis=0)) / spike.std(axis=0) # make unit variance so cov is identity
    MUAPs = generate_random_MUAPs(M=60, J=J, L=L)
    H = get_mixing_matrix(MUAPs, K=K)

    # Generate synthetic signal
    sig_out = get_sig_out(H, spike, L, K)
    # print(sig_out.shape)
    # plt.figure()
    # plt.plot(sig_out[30,:])
    # plt.show()

    # Get index of activity!
    ia = IA(sig_out, K=K)
    kernel = np.ones(L + K - 1)
    # kernel = signal.windows.hamming(L + K - 1)
    # ia_deconv, remainder = signal.deconvolve(ia, kernel)

    # STEPS FROM THE INTERNET
    # ia_deconv, remainder = signal.deconvolve(ia, kernel)
    # n = len(ia)-len(kernel)+1
    # s = int((len(ia)-n)/2)
    # deconv_res = np.zeros(len(ia))
    # deconv_res[s:len(ia)-s-1] = ia_deconv
    # deconv = deconv_res 

    plt.figure()
    plt.plot(ia)
    for idx in range(J):
        plt.plot(spike[:,idx])
    plt.show()

    # Plotting CST vs IA via PSD
    f, Pia = signal.welch(ia, fs, nperseg=1024, noverlap=512,)
    f, Pcst = signal.welch(spike.sum(axis=1), fs, nperseg=1024, noverlap=0)
    plt.figure()
    plt.plot(f, Pia)
    plt.plot(f, Pcst)
    plt.xlabel('Frequency (Hz)')
    plt.legend(['IA', 'CST'])
    plt.title('IA vs. CST')
    plt.show()

    # Comparing moving average with PSD of IA
    kernel = np.ones(L + K - 1) / (L + K - 1) # moving average filter!
    f, h = signal.freqz(a=1, b=kernel, fs=fs, worN=len(f))
    Pma = np.abs(h) ** 2
    Pma = Pma / Pma.sum()
    Pia = Pia / Pia.sum()
    plt.figure()
    plt.plot(f, Pia)
    plt.plot(f, Pma)
    plt.xlabel('Frequency (Hz)')
    plt.legend(['IA', 'Moving Average'])
    plt.show()


    # H = getH(data, J=40, K=1)
    # nrows = H.shape[0]
    # max_amp = H.max()
    # plt.figure()
    # for idx in range(nrows):
    #     plt.plot(H[idx, :] + nrows - 3*max_amp*idx)
    # plt.show()

    # # Plotting covariance of H
    # cov = np.cov(H) # across rows
    # fig, axs = plt.subplots(1, 1)
    # plt.title('Covariance of simulated matrix')
    # sns.heatmap(cov, ax=axs)
    # # sns.heatmap(np.cov(H, rowvar=False), ax=axs[1])
    # plt.show()


    # Run desired simulation with different parameters
    # data = [] # where to store each run

    # for run in tqdm(range(1, 11)):
    #     print('Run #{}...'.format(run)) 
    #     for K in tqdm([40, 10, 1]):
    #         print('For K = {}...'.format(K))
    #         for J in np.arange(1, 152, 20):

    #                 # Obtain singular values
    #                 H = getH(muap_data, K=K, J=J)
    #                 _, sigmas,_ = np.linalg.svd(H)
    #                 eigs = sigmas ** 2 # square singular vals to get eigenvalues
    #                 eigs = eigs / np.sum(eigs)

    #                 # Compute eigen cutoffs
    #                 eig1 =  eigen_cutoff(eigs, thrs=0.95) # accounting for 95% explained variance
    #                 eig2 = eigen_cutoff(eigs, thrs=0.99) # accounting for 99% explained variance

    #                 data.append([run, K, J, eig1, eig2])
    
    # columns = ['Run', 'K', 'J', 'eig1', 'eig2']
    # df = pd.DataFrame(data=data, columns=columns) # store simulation results in a dataframe
    # df.to_csv('sims.csv')

    # # RUNTIME CALCULATION
    # tf = time()
    # print('Experiment took {} s'.format(tf - t0))
    # print('Experiment took {} min'.format((tf - t0)/60))
    # print('Experiment took {} h'.format((tf - t0)/3600))