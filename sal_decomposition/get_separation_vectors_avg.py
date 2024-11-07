import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import scipy
from sklearn.cluster import KMeans
from MUEdit.processing_tools import extend_emg, whiten_emg, bandpass_filter, notch_filter, get_spikes, batch_process_filters, rate_of_agreement
from sal_decomposition.MUEdit.processing_tools import refine_mus, remove_outliers, remove_duplicates

# def get_ds_dictionaries(name, node):
  
#     fullname = node.name
#     if isinstance(node, h5py.Dataset):
#     # node is a dataset
#         print(f'Dataset: {fullname}; adding to dictionary')
#         ds_dict[fullname] = node
#         print('ds_dict size', len(ds_dict)) 
#     else:
#      # node is a group
#         print(f'Group: {fullname}; skipping')  
    
# with h5py.File('data.hdf5','r') as h5f:
        
#     ds_dict = {}  
#     print ('**Walking Datasets to get dictionaries**\n')
#     h5f.visititems(get_ds_dictionaries)
#     print('\nDONE')
#     print('ds_dict size', len(ds_dict))


def emanuele_grid_formatting(emg):
    ''' Takes grid, and based on the grid coordinate map, reorders the channels such that they can be made into a grid with the default reshaping settings.'''
    ElChannelMap = np.array([[53,56,57,59,33,2,3,4,16,8,6,14,254,246,247,248,255,250,252,226,218,193,196,206],
          [55,61,58,60,34,1,15,24,21,18,7,5,245,241,233,256,249,251,253,225,219,194,208,197],
          [54,42,45,48,38,25,27,32,22,19,10,13,244,234,237,240,230,227,221,217,220,195,207,198],
          [50,41,44,47,39,36,26,31,23,20,9,12,243,235,238,232,229,222,224,215,213,211,209,203],
          [51,49,43,46,40,37,35,28,29,30,17,11,242,236,239,231,228,223,216,214,212,210,201,202],
          [116,117,118,120,127,125,123,121,66,69,71,72,182,192,190,188,185,130,132,134,136,144,140,145],
          [106,114,119,128,126,124,122,65,67,70,79,78,178,181,191,189,186,129,131,133,135,143,141,147],
          [105,115,103,99,90,92,83,68,80,75,76,77,170,180,179,183,184,187,162,154,157,142,159,150],
          [97,108,109,100,101,93,94,84,86,88,81,73,168,173,175,169,177,165,167,153,156,139,160,149],
          [107,98,110,102,91,95,96,85,87,89,82,74,171,172,174,176,163,164,166,161,155,158,151,152]]).T-1

    emg = emg[ElChannelMap.ravel(), :] # should in priciple rearrange grid such that we can just use default reshape/ravel to get grid
    # # CHECK THAT EMG image makes sense!
    # import matplotlib.pyplot as plt
    # rms = np.std(emg_obj.signal_dict['data'], axis=1)
    # plt.figure()
    # plt.imshow(rms.reshape(24, 10))
    # plt.colorbar()
    # plt.savefig('rms_img_final.jpg')

    return emg

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
            pulse_trains_new, discharge_times_new = refine_mus(emg, pulse_trains.T, discharge_times_new, fsamp, extension_factor=extension_factor)
            discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, fsamp, cov_thr)
                

    return pulse_trains_new, discharge_times_new

def get_discharge_times(arr):
    ''' Based on discharge time format, get array of separation vectors'''
    distimes = []
    for grid_idx in range(arr['edition']['Distimeclean'].shape[0]):
        for mu_idx in range(arr[arr['edition']['Distimeclean'][grid_idx,0]].shape[0]):
            distimes.append(arr[arr[arr['edition']['Distimeclean'][grid_idx,0]][mu_idx,0]][:].ravel())
    return distimes

def grid_crop(segmented_data, xcrop: int, ycrop: int, grid_shape, fsamp):
    ''' Keep only a subgrid at the center, returning a signal of shape (H - 2ycrop, W - 2xcrop)'''
    uncropped_data = segmented_data
    # filter signal as it will be filtered later
    uncropped_data = notch_filter(uncropped_data, fsamp)
    uncropped_data = bandpass_filter(uncropped_data,fsamp,emg_type='surface')  

    emg_grid = segmented_data.reshape(grid_shape + (segmented_data.shape[1],))
    segmented_data = emg_grid[ycrop:emg_grid.shape[0]-ycrop, xcrop:emg_grid.shape[1]-xcrop, :].reshape(-1, emg_grid.shape[2])
    return uncropped_data, segmented_data

def open_decomp_output(DIR, name, temp_dir='./temp', MVC=20):
    ''' Open data from Simon's files so that it can be decomposed with Ciara's code.'''
    # arr = loadmat(os.path.join(DIR, name))
    with h5py.File(os.path.join(DIR, name), 'r') as arr:
        grid_idx = 0
        # Load acquisition signal
        signal = {}
        # signal['nchans'] = 64
        signal['fsamp'] = arr['signal']['fsamp'][0,0]
        signal['path'] = arr['signal']['path'][:].ravel()
        signal['target'] = arr['signal']['target'][:].ravel()
        signal['ied'] = arr['signal']['IED'][grid_idx, 0]
        # signal['electrode'] = arr['signal'][0,0][5][0, grid_idx][0]
        # signal['muscle'] = arr['signal'][0,0][6][0, grid_idx][0]
        # signal['nelectrodes'] = signal['nneedles'] + signal['ngrids']

        # Load data for all grids (right now only keep second grid)
        signal['data'] = arr['signal']['data'][:, :256]
        # pulse_trains = np.hstack([ arr[arr['edition']['Pulsetrainclean'][idx,0]][:] for idx in range(4) ])
        pulse_trains = np.hstack([ arr[arr['edition']['Pulsetrainclean'][idx,0]][:] for idx in range(4) ])
        signal['EMGmask'] = np.concatenate([arr[arr['signal']['EMGmask'][idx,0]][0,:] for idx in range(4) ])
        distimes = get_discharge_times(arr)

        decomp_dict = {} # initialising this dictionary here for later use
        mu_dict = dict(pulse_trains = None, discharge_times = [])# initialising a dictionary that is an empty nested list
        parameters = dict(arr['parameters'])

    return signal, pulse_trains, distimes, parameters, decomp_dict, mu_dict

if __name__ == '__main__':
    DIR ='/home/joao/Desktop/datasets/DatasetEmanuele/Subject2'
    name = 'S_2_30_1.otb+_decomp.mat_edited.mat'
    signal, pulse_trains, discharge_times, parameters_dict, decomp_dict, mu_dict = open_decomp_output(DIR, name)
    mu_count = pulse_trains.shape[1]

    # Filter EMG
    print('FILTERING EMG...')
    emg = bandpass_filter(signal['data'].T, int(signal['fsamp']))
    emg = notch_filter(emg, int(signal['fsamp']))

    # Filter out masked signals by zeroing them out
    emg[signal['EMGmask'].astype(bool), :] = 0.0
    emg = emanuele_grid_formatting(emg)
    uncropped_data, segmented_data = grid_crop()

    # Extend EMG
    print('LOADING, EXTENDING, WHITENING...')
    extension_factor = int(np.round(1000/emg.shape[0]))
    extended_emg = extend_emg(np.zeros((emg.shape[0]*extension_factor, emg.shape[1] + extension_factor - 1)) , emg, extension_factor)

    # Detrending & Whitening EMG
    extended_emg = scipy.signal.detrend(extended_emg, axis=- 1, type='constant', bp=0)
    whitened_emg, whitening_mat, dewhitening_mat = whiten_emg(extended_emg)

    # Average across spike times
    print('GETTING SEPARATION VECTORS BASED ON AVERAGE ACTIVITY AT SPIKE TIMES...')
    B = np.zeros((mu_count, whitened_emg.shape[0])) # initialize separation matrix
    for mu_idx in range(mu_count):
        bvec = whitened_emg[:, discharge_times[mu_idx].astype(int)].mean(axis=1) # assumes at least two spikes in each train
        B[mu_idx, :] = bvec / np.linalg.norm(bvec) # ensures they are unit vectors (i.e. normalized)

    # Test that it actually does work...
    print('GETTING PULSE ESTIMATES BASED ON SEPARATION VECS...')
    pulse_trains_est = B @ whitened_emg

    # Crop only to relevant path performed
    orig_sig_size = signal['data'].shape[0]
    plateau = np.where(signal['target'] >= max(signal['target'])*0.8)[0] # finding where the plateau is        
    segment = [plateau[0],plateau[-1]]

    whitened_emg = whitened_emg[:, int(segment[0]):int(segment[1])+extension_factor]
    uncropped_data = uncropped_data[:, int(segment[0]):int(segment[1])+extension_factor]
    segmented_data = segmented_data[:,int(segment[0]):int(segment[1])+extension_factor]
    pulse_trains = pulse_trains.T
    pulse_trains_crop = pulse_trains[:, int(segment[0]):int(segment[1]) + extension_factor]
    pulse_trains_est = pulse_trains_est[:, int(segment[0]):int(segment[1]) + extension_factor]

    # Get pulse trains via postprocessin
    pulse_trains_est, discharge_times_est = post_process_pulses(pulse_trains_est.T, emg, plateau=segment, fsamp=signal['fsamp'], extension_factor=extension_factor, cov_thr=0.5)
    
    corrs = np.zeros((pulse_trains.shape[0], pulse_trains_est.shape[0]))
    for rdx in tqdm(range(pulse_trains.shape[0])):
        for cdx in range(pulse_trains_est.shape[0]):
            stat, p = scipy.stats.pearsonr(pulse_trains[rdx, :], pulse_trains_est[cdx, :])
            corrs[rdx, cdx] = stat


    plt.figure()
    sns.heatmap(corrs)
    plt.title('Correlation Coefficients Between MUs')
    plt.show()

    # Plot some ground truth vs. pred
    for idx in range(3):
        # f_est, pxx_est = scipy.signal.welch(pulse_trains_est[0,:] - pulse_trains_est[0,:].mean(), int(signal['fsamp']))
        # f, pxx = scipy.signal.welch(pulse_trains[0,:] - pulse_trains[0,:].mean(), int(signal['fsamp']))
        plt.figure()
        # plt.plot(f_est, pxx_est, color='blue')
        # plt.plot(f, pxx, color='red', linestyle='--')
        plt.plot((pulse_trains_est[idx, :] - pulse_trains_est[idx, :].mean()) / (pulse_trains_est[idx, :].std() + 1e-9), color='blue')
        plt.plot((pulse_trains[idx, :] - pulse_trains[idx, :].mean()) / (pulse_trains[idx, :].std() + 1e-9) , color='red', linestyle='--')
        plt.legend(['Estimated', 'Ground Truth'])
        plt.show()

    # roas = rate_of_agreement2(discharge_times, discharge_times_est, threshold=int(0.0005*int(signal['fsamp'])))
    roas,_ = rate_of_agreement(discharge_times, pulse_trains, discharge_times_est, pulse_trains_est, maxlag=40, jitter_val=0.0005, fsamp=int(signal['fsamp']), duration=pulse_trains_est.shape[1])
    print(roas)
    
    ####################### SAVING DECOMPOSITION OUTPUT INTO DATAFRAMES ########################


    decomposition_dict = {'ied': signal['ied'], 'mu_filters': B, 'whiten_mat': whitening_mat,'pulse_trains': pulse_trains,
                        'discharge_times': discharge_times, 'original_data': signal['data'], 'filtered_data': emg_obj.signal_dict['segmented_data'],
                        'raw_sources': emg_obj.mu_dict['raw_sources'], 'uncropped_data': emg,
                        'path': emg_obj.signal_dict['path'], 'target': emg_obj.signal_dict['target'], 'plateau': emg_obj.plateau_coords, 'parameters': parameters_dict}

    # save dicitionaries into data file
    with open('./sal_decomposition/decomposition_data.pkl','wb') as file:
            pickle.dump(decomposition_dict, file)

    print('Decomposed data saved.')
