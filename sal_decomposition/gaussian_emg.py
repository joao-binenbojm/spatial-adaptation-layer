import numpy as np
import scipy
from tqdm import tqdm
from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def pca_whiten(signal):
    """
    Performs PCA whitening on the input signal and returns the whitened signal 
    and the whitening matrix used.

    Parameters:
    - signal: The input signal (shape: [n_samples, n_features])

    Returns:
    - whitened_signal: The whitened signal (shape: [n_samples, n_features])
    - whitening_matrix: The whitening matrix used for the transformation
    """
    signal = signal.T

    # Step 1: Perform PCA with whitening enabled
    pca = PCA(whiten=True)
    whitened_signal = pca.fit_transform(signal)

    # Step 2: The whitening matrix is simply the transformation matrix used by PCA
    whitening_matrix = pca.components_.T / np.sqrt(pca.explained_variance_)

    return whitened_signal.T, whitening_matrix

def generate_gaussian_muaps(H, W, L, fs, fx, fxmax):
    '''Generates gaussian sequence with a sampling rate freqfactor times greater than desired, so that we can lowpass and then downsample.
    '''
    muaps = np.random.normal(size=(mu_count, H, W, L))

    # Lowpass temporally to nyquist temporal frequency
    sos = scipy.signal.butter(N=3, Wn=1-1e-5, btype='low', output='sos')
    muaps = scipy.signal.sosfilt(sos, muaps, axis=3) # filter along rows

    # Lowpass signals to introduce smoothness
    sos = scipy.signal.butter(N=2, Wn=fxmax/(fx/2), btype='low', output='sos')
    muaps = scipy.signal.sosfilt(sos, muaps, axis=1) # filter along rows
    muaps = scipy.signal.sosfilt(sos, muaps, axis=2) # filter along columns
    return muaps

def generate_spike_trains(mu_count, duration, fs, Tmean=0.3, Tstd=0.1):
    ''' Generate motor unit spike trains.'''
    spts = np.zeros((mu_count, int(fs*duration)))
    dts = []
    # For every independent MU
    for mu_idx in range(mu_count):
        times = np.random.normal(loc=Tmean, scale=Tstd, size=int(duration/Tmean))
        times = np.cumsum(times) # cumulative sum of firing times
        dts.append(times) # become discharge times in seconds
        times = (fs*times[times <= duration]).astype(int)
        spts[mu_idx, times] = 1 # set all firing time values to 1
    
    return spts, dts

def generate_emg(spts, muaps):
    ''' Generate EMG based on spike trains and simulated MUAPs.'''
    EMG = np.zeros((muaps.shape[1], muaps.shape[2], spts.shape[1])) # empty EMG array
    for mdx in tqdm(range(muaps.shape[0])):
        for row in range(muaps.shape[1]):
            for col in range(muaps.shape[2]):
                EMG[row, col, :] += np.convolve(spts[mdx,:], muaps[mdx, row, col, :], mode='same')
    return EMG

def get_separation_vectors(muaps, R=None, xcrop=0, ycrop=0):
    ''' Based on MUAPs, just generate the separation vectors neccessary.'''
    N, H, W, L = muaps.shape
    if R is None: R = L
    Nch = (H-2*ycrop)*(W-2*xcrop)
    B = np.zeros((Nch*R, N))
    for mdx in tqdm(range(N)):
        for l in range(R):
            B[l*Nch:(l+1)*Nch, mdx] = muaps[mdx, ycrop:H-ycrop, xcrop:W-xcrop, R-l].ravel() # MUAP reversed is the separation vector itself!
        B[:, mdx] = B[:, mdx] / (np.linalg.norm(B[:, mdx]) + 1e-9) # make a unit vector
    return B

def grid_crop(emg, xcrop: int, ycrop: int):
    ''' Keep only a subgrid at the center, returning a signal of shape (H - 2ycrop, W - 2xcrop)'''
    cropped_emg = emg.copy() # ensures no aliasing issues
    # filter signal as it will be filtered later
    # emg_obj.signal_dict['uncropped_data'] = notch_filter(emg_obj.signal_dict['uncropped_data'],emg_obj.signal_dict['fsamp'])
    # emg_obj.signal_dict['uncropped_data'] = bandpass_filter(emg_obj.signal_dict['uncropped_data'],emg_obj.signal_dict['fsamp'],emg_type = emg_obj.emgopt)  

    cropped_emg = cropped_emg[ycrop:cropped_emg.shape[0]-ycrop, xcrop:cropped_emg.shape[1]-xcrop, :]
    return cropped_emg

fs = 2048 # Hz
Tx, Ty = 0.01, 0.01 # 1cm IED
fx, fy = 1/Tx, 1/Ty
L = 50 # 50 samples in a MUAP
mu_count = 20

# Muap grid
H, W = 25, 10 # H*W electrodes in a grid
xcrop, ycrop = 2,2
# muaps = np.random.normal(size=(mu_count, H, W, L))
muaps = generate_gaussian_muaps(H, W, L, fs, fx, fxmax=49.9)

# Spike trains
duration = 10 # 10s
spts, dts = generate_spike_trains(mu_count, duration, fs)
uncropped_emg = generate_emg(spts, muaps)
uncropped_emg = uncropped_emg + np.random.normal(loc=0, scale=0, size=uncropped_emg.shape)
# uncropped_emg = (uncropped_emg - uncropped_emg.mean()) / (uncropped_emg.std() + 1e-12)
# uncropped_emg = EMG.copy()
EMG = grid_crop(uncropped_emg, xcrop, ycrop)

# Test separation vector!
# R = muaps.shape[-1]
R = 16
extended_emg = np.zeros((EMG.shape[0]*EMG.shape[1]*R, EMG.shape[2] + R - 1)) # create extended EMG template
print('EXTENDING EMG...')
extended_emg = extend_emg(extended_emg, EMG.reshape(EMG.shape[0]*EMG.shape[1], EMG.shape[2]), R) # extend EMG signal
whitened_emg, whitening_mat, dewhitening_mat = whiten_emg(extended_emg)
print('WHITENING EMG...')
# whitened_emg, whitening_mat = pca_whiten(extended_emg)
# whitened_emg = whitening_mat.astype(np.float32) @ extended_emg.astype(np.float32)
# B = get_separation_vectors_avg(extended_emg, spts)
# whitened_emg = extended_emg
B = get_separation_vectors(muaps, R=R, xcrop=xcrop, ycrop=ycrop)
B = (whitening_mat @ B).T # apply transpose of whitening mtrix to separation vectors, such that we still get spikes from whitened obvs.
mu1 = (B[[0], :] @  whitened_emg).ravel()
plt.figure()
plt.plot(mu1 / mu1.max())
plt.plot(spts[0,:]/spts.max())
plt.legend(['estimated', 'original'])
# plt.savefig('mu')
plt.show()

####################### SAVING DECOMPOSITION OUTPUT INTO DATAFRAMES ########################

# Creating a decomposition dictionary as output
parameters_dict = {
    'file_path': '', 'n_its':  0, 'ref_exist': 0, 'fsamp': fs,
    'target_thres': 0.8, 'nwins': 0, 'check_EMG': 0,
    'drawing_mode': 0, 'differential_mode': 0, 'peel_off': 1,
    'refine_MU': 1, 'sil_thr': 0.9, 'ext_factor': R,
    'edges': [], 'dup_thr': 0.3, 'cov_filter': .5, 
    'cov_thr': 0.5, 'xcrop': xcrop, 'ycrop': ycrop, 'snr': None}


decomposition_dict = {'chan_name': '','muscle_name': '',
                    'ied': Tx, 'mu_filters': B, 'whiten_mat': whitening_mat,
                    'inv_extend_obvs': 0 ,'pulse_trains': [spts],
                    'discharge_times': dts, 'original_data': uncropped_emg.reshape(-1, uncropped_emg.shape[2]), 
                    'filtered_data': EMG.reshape(-1, EMG.shape[2]), 'uncropped_data': uncropped_emg.reshape(-1, uncropped_emg.shape[2]),
                    'path': 0, 'target': 0, 'plateau': [0,-1], 'parameters': parameters_dict}

# save dicitionaries into data file
with open('./sal_decomposition/decomposition_data.pkl','wb') as file:
        pickle.dump(decomposition_dict, file)

print('Decomposed data saved.')