import os
from time import time
from copy import deepcopy
from math import floor

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
from scipy import signal
import torch
import wfdb
import matplotlib.pyplot as plt

from emg_processing import bandpass, bandstop, identity, get_rms_signal
from networks_utils import median_pool_2d

## TEMPORARY UTILS FUNCTION LOCATIONS ##

def process_binary_signal(input_signal, min_length=1000, edge_trim=500):
    input_signal = np.array(input_signal)
    output = np.copy(input_signal)

    # Find start and end indices of 1-segments
    padded = np.pad(input_signal, (1, 1), 'constant')
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        length = end - start
        if length < min_length:
            output[start:end] = 0  # Zero out the short segment
        else:
            # Trim edges
            trim_start = min(edge_trim, length // 2)
            trim_end = min(edge_trim, length // 2)
            output[start:start + trim_start] = 0
            output[end - trim_end:end] = 0

    return output

def upsample_signal(input_signal, up=2048, down=100):
    """
    Upsample a signal by a factor of up and downsample by a factor of down.
    """
    # Resample the signal
    gcd = np.gcd(up, down)
    up //= gcd
    down //= gcd
    resampled_signal = signal.resample_poly(input_signal, up, down)
    return resampled_signal

########################################################################



class EMGData:
    
    def __init__(self, dataset='csl', path='../datasets/capgmyo/dbb_csl', sub='subject1', transform=None, target_transform=None, norm=0,
                  num_gestures=26, num_repetitions=10, input_shape=(8, 24), fs=2048, rep_duration=None, Trms=0.25, sessions='session1', 
                  intrasession=False, rms=True, remove_baseline='mean-square', median_filter=False, gest_subset=None, is_segment=True):
        # Store all appropriate data parameters
        self.dataset = dataset
        self.path = path
        self.fs = fs
        self.Mrms = int(self.fs*Trms)
        self.rms = rms     
        self.intrasession = intrasession
        if rep_duration is not None:
            self.num_samples = int(fs*rep_duration) # number of samples per repetition
        else:
            self.num_samples = fs
        self.norm = norm
        if gest_subset is None:
            self.gest_subset = list(range(num_gestures))
            self.num_gestures = num_gestures
        else:
            self.num_gestures = len(gest_subset)
            self.gest_subset = gest_subset
        self.num_repetitions = num_repetitions
        self.input_shape = input_shape
        self.sessions = sessions
        self.num_sessions = len(sessions)
        self.sub = sub
        self.current_session = 0 # to keep track of what session we are extracting from
        self.remove_baseline=remove_baseline
        self.median_filter = median_filter
        self.is_segment = is_segment

        # Mask that determines which EMG segments are active
        self.active = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples), dtype=np.bool_)
        self.durations = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions)) # durations of gesture segments

        # Preinitialize Data tensors
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

        # Target transforms
        self.transform = transform
        self.target_transform = target_transform

    def apply_median_filter(self, X):
        '''Apply median filtering depending on the specific dataset.'''
        if self.dataset == 'csl' or self.dataset == 'capgmyo':
            X = median_pool_2d(X)
        elif self.dataset == 'hyser':
            Xtop, Xbot = X[:, :, :X.shape[2]//2, :], X[:, :, X.shape[2]//2:, :]
            Xtop, Xbot = median_pool_2d(Xtop), median_pool_2d(Xbot)
            X = torch.cat((Xtop, Xbot), dim=2)
        elif 'grabmyo' in self.dataset:
            X = median_pool_2d(X, kernel_size=(2,3), padding=0, circular=True)
        return X

    def get_images(self, emg_segment):
        ''' Takes in either raw sEMG or RMS activity and returns it in the appropriate shape given the dataset being used.'''
        if self.dataset == 'csl':
            images = np.flip(np.array(emg_segment).reshape(emg_segment.shape[0], 1, 8, 24, order='F'), axis=0)
            images = images[:, :, 1:, :] # drop first row given bipolar nature of data and create list

        elif self.dataset == 'capgmyo':
            images = np.array(emg_segment).reshape(emg_segment.shape[0], 1, self.input_shape[0], self.input_shape[1], order='F')

        elif self.dataset == 'hyser':
            ngrids = 4
            subimages = []
            for grid_idx in range(ngrids):
                subimage = np.array(emg_segment[:, grid_idx*64:(grid_idx+1)*64]).reshape(emg_segment.shape[0], 1, 8, 8)
                subimage = np.flip(np.flip(subimage, axis=2), axis=3)
                subimages.append(subimage)
            images = np.concatenate(subimages, axis=2) # append along horizontal direction
                
        elif 'grabmyo' in self.dataset: # can either be for the forearm or wrist
            T = emg_segment.shape[0]
            if 'forearm' in self.dataset:
                images = emg_segment[:,:16].reshape(T, 1, 2, 8)
            elif 'wrist' in self.dataset:
                images = emg_segment[:,16:].reshape(T, 1, 2, 6)
            # images = np.concatenate((forearm, wrist), axis=3) # concats grids as a horizontally long array
        else:
            raise Exception("No dataset specified.")
        return images 
        
    def get_baseline(self, DIR):
        '''Given the subject/session directory, computes the baseline activity for every channel, depending on the given dataset.'''
        if self.dataset == 'csl':
            mat = sio.loadmat(os.path.join(DIR, 'gest0.mat'))
            reps = mat['gestures'].shape[0] # number of repetitions
            baseline = np.zeros((1, 192))
            baseline_samp_count = 0
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = emg - emg.mean(axis=0, keepdims=True)
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                emg = emg**2
                baseline += emg.sum(axis=0, keepdims=True)
                baseline_samp_count += emg.shape[0]
            baseline = baseline/baseline_samp_count
            if self.remove_baseline == 'root-mean-square':
                baseline = np.sqrt(baseline)

        elif self.dataset == 'capgmyo':
            filenames = os.listdir(DIR)
            baseline = np.zeros((1, np.prod(self.input_shape)))
            baseline_samp_count = 0
            
            cur_rec_id = str(int(self.sub.replace('subject',''))*2 + self.current_session + 1) # current recording ID
            id_len = len(cur_rec_id)
            for idx in range(3-id_len): cur_rec_id = '0' +  cur_rec_id
            filenames = [file for file in filenames if file[:3] == cur_rec_id] # filter to this subject/session
            filenames = [file for file in filenames if ('100' not in file) and ('101' not in file)] # removes MVC recordings
            for gdx, name in enumerate(filenames):
                mat = sio.loadmat(os.path.join(DIR, name))
                emg, labels = mat['data'], mat['gesture'].ravel()
                emg = emg - emg[labels==0,:].mean(axis=0, keepdims=True)
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                emg = emg**2

                baseline += emg.sum(axis=0, keepdims=True)
            baseline / baseline_samp_count
            if self.remove_baseline == 'root-mean-square':
                baseline = np.sqrt(baseline)

        elif self.dataset == 'hyser':
            DIR = DIR.replace('pr_dataset', 'mvc_dataset')
            baseline = np.zeros((1, 256))
            # baseline_samp_count = 0
            # for finger_idx in range(5):
            #     for movement in ['extension', 'flexion']:
            #         # Process force to get rest periods
            #         force_record = wfdb.rdrecord(os.path.join(DIR, f"mvc_force_finger{finger_idx+1}_{movement}"))
            #         force = np.abs(force_record.p_signal.mean(axis=1))
            #         force = upsample_signal(force, up=self.fs, down=100)
            #         force = (force - np.min(force)) / (np.max(force) - np.min(force))                    
            #         force_off = (force < 0.2*force.max()).astype(int)
            #         force_off_processed = process_binary_signal(force_off)
                    
            #         # Process EMG and get baseline from rest segments
            #         emg_record = wfdb.rdrecord(os.path.join(DIR, f"mvc_raw_finger{finger_idx+1}_{movement}"))
            #         emg = emg_record.p_signal - emg_record.p_signal.mean(axis=0, keepdims=True) # make zero mean signals
            #         emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
            #         emg = emg[force_off_processed.astype(np.bool_), :] # keep only rest segments
            #         baseline += (emg**2).sum(axis=0)
            #         baseline_samp_count += emg.shape[0]
            
            # baseline = baseline / baseline_samp_count
        
        elif 'grabmyo' in self.dataset:
            baseline = np.zeros((1, 28))
            baseline_samp_count = 0
            subdir = os.path.basename(DIR) # get subject_dir
            for idx in range(self.num_repetitions):
                fname = f"{subdir}_gesture17_trial{idx+1}" # get given trial 
                record = wfdb.rdrecord(os.path.join(DIR, fname))
                emg = record.p_signal
                emg = emg - emg.mean(axis=0, keepdims=True)
                keep_channels = ['U' not in name for name in record.sig_name] # drop redundant channels
                emg = emg[:, keep_channels]
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                center = emg.shape[0] // 2
                emg = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                emg_square = emg**2
                baseline += emg_square.sum(axis=0, keepdims=True)
                baseline_samp_count += emg.shape[0]
                # images = self.get_images(rms)
                # images = images.reshape(1, 1, *images.shape)
                # baseline += images.mean(axis=2, keepdims=True)
            baseline = baseline/ baseline_samp_count
            if self.remove_baseline == 'root-mean-square':
                baseline = np.sqrt(baseline)

        else:
            raise Exception("No dataset specified.")
        return baseline
    
    def segment(self, emg, baseline):
        ''' Segments a given EMG repetition based on CSL segmentation algorithm.'''
        ksize, stride = int(0.0732*self.fs), int(0.0732*self.fs) # getting samples from fixed number of seconds

        # Get RMS
        emg = emg.T
        emg_tensor = torch.tensor(emg).view(emg.shape[0], 1, emg.shape[1]) # convert to PyTorch for strided convolution functionality
        weight = torch.ones(1, 1, ksize, dtype=torch.float64) / ksize # moving average filter
        ms = torch.nn.functional.conv1d(emg_tensor**2, weight, stride=stride)
        rms = torch.sqrt(ms).view(emg.shape[0], -1).T # convert to original shape (but of different length after conv.)

        # Remove baseline and apply median filter
        rms = rms - baseline
        bs_imgs = self.get_images(rms).copy()
        bs_imgs = median_pool_2d(torch.tensor(bs_imgs), kernel_size=(3, 1), padding=(1, 0)) # vertical median pooling, along muscle fiber direction

        # Compute threshold and threshold images
        sum_rms = bs_imgs.sum(dim=(1, 2, 3)) # sum of RMS values of all channels for each given window
        thrs = sum_rms.mean() # average summed RMS across windows
        active = np.array(sum_rms > thrs) # get windows that are active
        active = median_filter(active, size=3, mode='nearest') # doesn't remove the first and last active sub-segment
 
        # Remove all segments found but the longest, and return the start and end in terms of original sampling rate
        changes = np.diff(active, prepend=0)
        start_indices, end_indices = np.where(changes > 0)[0], np.where(changes < 0)[0]

        # If segment begins or ends active
        if len(start_indices) == 0: start_indices = np.array([0])
        if len(end_indices) == 0: end_indices = np.array([len(active) - 1])

        min_len = min(len(start_indices), len(end_indices))
        start_indices, end_indices = start_indices[:min_len], end_indices[:min_len]
        max_idx = np.argmax(end_indices - start_indices)
        start, end = start_indices[max_idx], end_indices[max_idx]

        # Obtain start and end in samples in terms of original sampling rate
        start, end = start*stride, end*stride

        return start, end

    def extract_frames(self, DIR=None):
        ''' Placeholder function to be overriden by child classes.'''
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        return X, Y

    def load_tensors(self):
        ''' Takes in data files and cretes complete data tensor for either intrasession or intersession case.'''
        for idx, session in enumerate(self.sessions):
            # DIR = os.path.join(self.path, self.sub, session)
            DIR = self.path
            Xs, Ys = self.extract_frames(DIR)
            self.X[idx, :, :, :, :, :, :] = Xs # add data extracted from given session
            self.Y[idx, :, :, :] = Ys # add labels extracted from given session
            self.current_session += 1

        # Convert data to tensor
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)
    
    def get_tensors_intrasession(self, session, rep_idx=None):
        """
        Logic for intrasession case.
        """
        idxs = list(range(self.num_repetitions))
        test_idx = [idxs.pop(rep_idx)]

        # Get train and test splits
        X_train = self.X[[session], :, idxs, :, :, :, :]
        Y_train = self.Y[[session], :, idxs, :]
        X_test = self.X[[session], :, test_idx, :, :, :, :]
        Y_test = self.Y[[session], :, test_idx, :]

        if self.is_segment:
            # Segmentation-specific logic
            train_active = self.active[[session], :, idxs, :]
            test_active = self.active[[session], :, test_idx, :]
            X_train, Y_train = X_train[torch.tensor(train_active)], Y_train[torch.tensor(train_active)]
            X_test, Y_test = X_test[torch.tensor(test_active)], Y_test[torch.tensor(test_active)]
            test_durations = self.durations[session, :, test_idx]
        else:
            # Standard logic
            test_durations = self.num_samples * np.ones(self.Y.shape[1])

        # Convert to torch tensors of type float32
        X_train, X_test = X_train.to(torch.float32), X_test.to(torch.float32)

        X_train, Y_train = torch.flatten(X_train, end_dim=-4), torch.flatten(Y_train, end_dim=-1)
        X_test, Y_test = torch.flatten(X_test, end_dim=-4), torch.flatten(Y_test, end_dim=-1)

        if self.median_filter:
            print('APPLYING MEDIAN FILTER...')
            X_train, X_test = self.apply_median_filter(X_train), self.apply_median_filter(X_test)
        elif 'grabmyo' in self.dataset:
            X_train = X_train.mean(dim=2, keepdim=True)
            X_test = X_test.mean(dim=2, keepdim=True)

        return X_train, Y_train, X_test, Y_test, test_durations.ravel()

    def get_tensors_intersession(self, train_session, test_session, rep_idx=None, gest_idxs=None):
        """
        Logic for intersession case.
        """
        idxs = list(range(self.num_repetitions))
        adapt_idx = []

        if rep_idx is not None:
            if isinstance(rep_idx, int):
                rep_idx = [rep_idx]
            for one_rep_idx in sorted(rep_idx, reverse=True):
                adapt_idx.append(idxs.pop(one_rep_idx))
        else:
            adapt_idx = idxs

        # Get train, adapt, and test splits
        X_train = self.X[[train_session], :, :, :, :, :, :]
        Y_train = self.Y[[train_session], :, :, :]
        X_adapt = self.X[[test_session], :, adapt_idx, :, :, :, :]
        Y_adapt = self.Y[[test_session], :, adapt_idx, :]
        X_test = self.X[[test_session], :, idxs, :, :, :, :]
        Y_test = self.Y[[test_session], :, idxs, :]

        if gest_idxs is not None:
            if isinstance(gest_idxs, int):
                gest_idxs = [gest_idxs]
            X_adapt = X_adapt[:, gest_idxs, :, :, :, :]
            Y_adapt = Y_adapt[:, gest_idxs, :]

        if self.is_segment:
            # Segmentation-specific logic
            train_active = self.active[[train_session], :, :, :]
            adapt_active = self.active[[test_session], :, adapt_idx, :]
            test_active = self.active[[test_session], :, idxs, :]
            X_train, Y_train = X_train[torch.tensor(train_active)], Y_train[torch.tensor(train_active)]
            X_adapt, Y_adapt = X_adapt[torch.tensor(adapt_active)], Y_adapt[torch.tensor(adapt_active)]
            X_test, Y_test = X_test[torch.tensor(test_active)], Y_test[torch.tensor(test_active)]
            adapt_durations = self.durations[test_session, :, adapt_idx]
            test_durations = self.durations[test_session, :, idxs]
        else:
            # Standard logic
            # adapt_durations = self.num_samples * np.ones(self.Y.shape[1])
            test_durations = self.num_samples * np.ones(self.Y.shape[1])

        # Convert to torch tensors of type float32
        X_train, X_adapt, X_test = X_train.to(torch.float32), X_adapt.to(torch.float32), X_test.to(torch.float32)

        X_train, Y_train = torch.flatten(X_train, end_dim=-4), torch.flatten(Y_train, end_dim=-1)
        X_adapt, Y_adapt = torch.flatten(X_adapt, end_dim=-4), torch.flatten(Y_adapt, end_dim=-1)
        X_test, Y_test = torch.flatten(X_test, end_dim=-4), torch.flatten(Y_test, end_dim=-1)

        if self.median_filter:
            print('APPLYING MEDIAN FILTER...')
            X_train, X_adapt, X_test = self.apply_median_filter(X_train), self.apply_median_filter(X_adapt), self.apply_median_filter(X_test)
        elif 'grabmyo' in self.dataset:
            X_train = X_train.mean(dim=2, keepdim=True)
            X_test = X_test.mean(dim=2, keepdim=True)
            X_adapt = X_adapt.mean(dim=2, keepdim=True)
        
        # # HYSER TEST PLOTTING
        # plt.figure()
        # fig, ax = plt.subplots(2, 6)
        # # vmin, vmax = X_train.min(), X_train.max()
        # for idx in range(2):
        #     for jdx in range(6):
        #         label = idx*6 + jdx
        #         ax[idx, jdx].imshow(X_train[Y_train==label,0,:,:].mean(dim=0))
        #         ax[idx, jdx].axis('off')
        #         ax[idx, jdx].set_title(f'Label: {label}')
        
        # plt.savefig('baseline')

        return X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations.ravel()

    def get_tensors_simulation(self, session, adapt_rep_idx, test_rep_idx):
        """
        Logic for simulation case.
        """
        idxs = list(range(self.num_repetitions))
        if self.dataset == 'hyser':
            adapt_idx = [idxs.pop(adapt_rep_idx)]
            test_idx = idxs.copy() # make test set a copy of training idxs for Hyser sims
        else:
            adapt_idx = [idxs.pop(adapt_rep_idx)]  # Remove adapt repetition from train indices
            test_idx = adapt_idx.copy()  # Remove test repetition from train indices
            
        # Get train, adapt, and test splits
        X_train = self.X[[session], :, idxs, :, :, :, :]
        Y_train = self.Y[[session], :, idxs, :]
        X_adapt = self.X[[session], :, adapt_idx, :, :, :, :]
        Y_adapt = self.Y[[session], :, adapt_idx, :]
        X_test = self.X[[session], :, test_idx, :, :, :, :]
        Y_test = self.Y[[session], :, test_idx, :]

        if self.is_segment:
            # Segmentation-specific logic
            train_active = self.active[[session], :, idxs, :]
            adapt_active = self.active[[session], :, adapt_idx, :]
            test_active = self.active[[session], :, test_idx, :]
            X_train, Y_train = X_train[torch.tensor(train_active)], Y_train[torch.tensor(train_active)]
            X_adapt, Y_adapt = X_adapt[torch.tensor(adapt_active)], Y_adapt[torch.tensor(adapt_active)]
            X_test, Y_test = X_test[torch.tensor(test_active)], Y_test[torch.tensor(test_active)]
            # adapt_durations = self.durations[session, :, adapt_idx]
            test_durations = self.durations[session, :, test_idx]
        else:
            # Standard logic
            # adapt_durations = self.num_samples * np.ones(self.Y.shape[1])
            test_durations = self.num_samples * np.ones(self.Y.shape[1])

        # Convert to torch tensors of type float32
        X_train, X_adapt, X_test = X_train.to(torch.float32), X_adapt.to(torch.float32), X_test.to(torch.float32)

        X_train, Y_train = torch.flatten(X_train, end_dim=-4), torch.flatten(Y_train, end_dim=-1)
        X_adapt, Y_adapt = torch.flatten(X_adapt, end_dim=-4), torch.flatten(Y_adapt, end_dim=-1)
        X_test, Y_test = torch.flatten(X_test, end_dim=-4), torch.flatten(Y_test, end_dim=-1)
        
        # # TESTING OUR BASELINE ACTVIATION
        # if self.dataset == 'hyser' and self.rms:
        #     # X_train = X_train - X_train.mean(dim=[0,1], keepdim=True) # remove baseline activity
        #     # X_adapt = X_adapt - X_adapt.mean(dim=[0,1], keepdim=True)
        #     # X_test = X_test - X_test.mean(dim=[0,1], keepdim=True)
        #     X_train = X_train**2 # Get MS from RMS
        #     X_train = X_train - X_train.mean(dim=[0,1], keepdim=True) # remove baseline activity
        #     X_train[X_train < 0] = 0 # ensure all values are positive
        #     X_train = torch.sqrt(X_train) # Get RMS from MS
        #     X_adapt = X_adapt**2 # Get MS from RMS
        #     X_adapt = X_adapt - X_adapt.mean(dim=[0,1], keepdim=True)
        #     X_adapt[X_adapt < 0] = 0 # ensure all values are positive
        #     X_adapt = torch.sqrt(X_adapt)
        #     X_test = X_test**2 # Get MS from RMS
        #     X_test = X_test - X_test.mean(dim=[0,1], keepdim=True)
        #     X_test[X_test < 0] = 0 # ensure all values are positive
        #     X_test = torch.sqrt(X_test)

        if self.median_filter:
            print('APPLYING MEDIAN FILTER...')
            X_train, X_adapt, X_test = self.apply_median_filter(X_train), self.apply_median_filter(X_adapt), self.apply_median_filter(X_test)
        elif 'grabmyo' in self.dataset:
            X_train = X_train.mean(dim=2, keepdim=True)
            X_test = X_test.mean(dim=2, keepdim=True)
            X_adapt = X_adapt.mean(dim=2, keepdim=True)

        return X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test,  test_durations.ravel()

    def oversample_repetitions(self, X, Y, cur_label, reps, missing):
        ''' Used when there is a non-uniform number of repetitions across gestures for a given uer.
            Here, we oversample previous repetitions.
        '''
        for idx in range(reps, reps+missing):
            rep_idx = np.random.randint(0, reps)
            Xsample = X[cur_label, rep_idx, :, :, :, :]
            X[cur_label, idx, :, :, :, :] = Xsample
            Ysample = Y[cur_label, rep_idx, :]
            Y[cur_label, idx, :] = Ysample
        return X, Y


# class EMGSegmentData(EMGData):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def segment(self, emg, baseline):
#         ''' Segments a given EMG repetition based on CSL segmentation algorithm.'''
#         ksize, stride = int(0.0732*self.fs), int(0.0732*self.fs) # getting samples from fixed number of seconds

#         # Get RMS
#         emg = emg.T
#         emg_tensor = torch.tensor(emg).view(emg.shape[0], 1, emg.shape[1]) # convert to PyTorch for strided convolution functionality
#         weight = torch.ones(1, 1, ksize, dtype=torch.float64) / ksize # moving average filter
#         ms = torch.nn.functional.conv1d(emg_tensor**2, weight, stride=stride)
#         rms = torch.sqrt(ms).view(emg.shape[0], -1).T # convert to original shape (but of different length after conv.)

#         # Remove baseline and apply median filter
#         images = self.get_images(rms)
#         baseline = baseline[0,0,:,:,:,:] # remove first two singleton dimensions for easier baseline subtraction
#         bs_imgs = images - baseline # remove baseline activity from the images
#         bs_imgs = median_pool_2d(torch.tensor(bs_imgs), kernel_size=(3, 1), padding=(1, 0)) # vertical median pooling, along muscle fiber direction
        
#         # Compute threshold and threshold images
#         sum_rms = bs_imgs.sum(dim=(1, 2, 3)) # sum of RMS values of all channels for each given window
#         thrs = sum_rms.mean() # average summed RMS across windows
#         active = np.array(sum_rms > thrs) # get windows that are active
#         active = median_filter(active, size=3, mode='nearest') # doesn't remove the first and last active sub-segment
 
#         # Remove all segments found but the longest, and return the start and end in terms of original sampling rate
#         changes = np.diff(active, prepend=0)
#         start_indices, end_indices = np.where(changes > 0)[0], np.where(changes < 0)[0]

#         # If segment begins or ends active
#         if len(start_indices) == 0: start_indices = np.array([0])
#         if len(end_indices) == 0: end_indices = np.array([len(active) - 1])

#         min_len = min(len(start_indices), len(end_indices))
#         start_indices, end_indices = start_indices[:min_len], end_indices[:min_len]
#         max_idx = np.argmax(end_indices - start_indices)
#         start, end = start_indices[max_idx], end_indices[max_idx]

#         # Obtain start and end in samples in terms of original sampling rate
#         start, end = start*stride, end*stride

#         return start, end

#     def get_tensors(self, train_session=None, test_session=None, rep_idx=None, gest_idxs=None):
#         ''' Return data in desired format of surface images, with a leave-one-out approach for testing.
#         '''
#         if self.intrasession:
#             idxs = list(range(self.num_repetitions))
#             test_idx = idxs.pop(rep_idx)

#             # Get appropriate train/test/adapt split
#             X_train = self.X[[test_session], :, idxs, :, :, :, :]
#             Y_train = self.Y[[test_session], :, idxs, :]
#             train_active = self.active[[test_session], :, idxs, :]
#             X_test = self.X[[test_session], :, [test_idx], :, :, :, :]
#             Y_test = self.Y[[test_session], :, [test_idx], :]
#             test_active = self.active[[test_session], :, [test_idx], :]

#             # Get only detected active segments of activity
#             X_train, Y_train = X_train[torch.tensor(train_active)], Y_train[torch.tensor(train_active)]
#             X_test, Y_test = X_test[torch.tensor(test_active)], Y_test[torch.tensor(test_active)]
#             test_durations = self.durations[test_session, :, test_idx]         
            
#             # Convert to torch tensors of type float32
#             X_train, X_test = X_train.to(torch.float32), X_test.to(torch.float32)
#             return X_train, Y_train, X_test, Y_test, test_durations
        
#         else:
#             idxs = list(range(self.num_repetitions))
#             # If fine-tuning on a single repetition
#             if rep_idx is not None:
#                 adapt_idx = [idxs.pop(rep_idx)]
#             else: # else, fine-tune on all available test data
#                 adapt_idx = idxs

#             # Get appropriate train/test/adapt split
#             X_train = self.X[[train_session], :, :, :, :, :, :]
#             Y_train = self.Y[[train_session], :, :, :]
#             train_active = self.active[[train_session], :, :, :]
#             X_adapt = self.X[[test_session], :, adapt_idx, :, :, :, :]
#             Y_adapt = self.Y[[test_session], :, adapt_idx, :]
#             adapt_active = self.active[[test_session], :, adapt_idx, :]
#             X_test = self.X[[test_session], :, idxs, :, :, :, :]
#             Y_test = self.Y[[test_session], :, idxs, :]
#             test_active = self.active[[test_session], :, idxs, :]

#             # If fine-tuning on a single repetition of one or few gestures
#             if gest_idxs is not None:
#                 if isinstance(gest_idxs, int): gest_idxs = [gest_idxs] # if single gesture for calibration selected
#                 X_adapt = X_adapt[:, gest_idxs, :, :, :, :] # only keep the gestures for calibration
#                 Y_adapt = Y_adapt[:, gest_idxs, :]
#                 adapt_active = adapt_active[:, gest_idxs, :]

#             # Get only detected active segments of activity
#             X_train, X_adapt, X_test = X_train[torch.tensor(train_active)], X_adapt[torch.tensor(adapt_active)], X_test[torch.tensor(test_active)]
#             Y_train, Y_adapt, Y_test = Y_train[torch.tensor(train_active)], Y_adapt[torch.tensor(adapt_active)], Y_test[torch.tensor(test_active)]
#             test_durations = self.durations[test_session, :, idxs]

#             # Convert to torch tensors of type float32
#             X_train, X_adapt, X_test = X_train.to(torch.float32), X_adapt.to(torch.float32), X_test.to(torch.float32)
#             return X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations.ravel()

#     def oversample_repetitions(self, X, Y, cur_label, reps, missing):
#         ''' Used when there is a non-uniform number of repetitions across gestures for a given uer.
#             Here, we oversample previous repetitions.
#         '''
#         for idx in range(reps, reps+missing):
#             rep_idx = np.random.randint(0, reps)
#             Xsample = X[cur_label, rep_idx, :, :, :, :]
#             X[cur_label, idx, :, :, :, :] = Xsample
#             Ysample = Y[cur_label, rep_idx, :]
#             Y[cur_label, idx, :] = Ysample
#             dur_sample = self.durations[self.current_session, cur_label, rep_idx]
#             self.durations[self.current_session, cur_label, idx] = dur_sample
#             active_sample = self.active[self.current_session, cur_label, rep_idx, :]
#             self.active[self.current_session, cur_label, idx, :] = active_sample

#         return X, Y

############################################################## CAPGMYO EMG TENSORIZERS #####################################################################


class CapgmyoData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session from capgmyo.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        print('ESTIMATING BASELINE ACTIVITY...')
        if self.remove_baseline or self.is_segment:
            baseline = self.get_baseline(DIR)

        cur_rec_id = int(self.sub.replace('subject',''))*2 + self.current_session + 1 # current recording ID
        for file in filenames:
            rec_id, gest = file.split('-')
            rec_id, gest = int(rec_id.lstrip('0')), int(gest.replace('.mat', '').lstrip('0'))

            if gest == 100 or gest == 101: # skip MVC recordings
                continue
            if cur_rec_id != rec_id: # skip recordings not from same subject/session
                continue

            # If a gesture, load file appropriately
            mat = sio.loadmat(os.path.join(DIR, file))
            emg = mat['data']
            emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
            if self.rms:
                emg = get_rms_signal(emg, Mrms=self.Mrms) # get RMS
                if self.remove_baseline:
                    if self.remove_baseline == 'root-mean-square':
                        emg = emg - baseline
                    elif self.remove_baseline == 'mean-square':
                        emg_square = emg**2
                        emg_square = emg_square - baseline
                        emg_square[emg_square < 0] = 0 # ensure all values are positive
                        emg = np.sqrt(emg_square) # get RMS from MS

            cur_label = gest-1

            # Account for exception case of missing repetitions
            labels = mat['gesture'].ravel()
            labels_rolled = np.roll(np.copy(labels), 1)
            delta = (labels - labels_rolled)
            indices = np.where(delta != 0)[0]
            if len(indices) > 20:
                indices = indices[:20]
            reps = len(indices) // 2 # number of repetitions is equivalent to half of the number of changepoints
            missing = self.num_repetitions - reps # number of missing repetitions from the protocol

            # For each repetition available
            for idx in range(0, len(indices), 2):
                start, end = indices[idx], indices[idx+1]
                center = (start + end) // 2 # get the central index of the given repetition
                emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = self.get_images(emg_segment)
                

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx//2, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx//2, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
        
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

        # # Remove baseline activity
        # if self.rms:
        #     if self.remove_baseline:
        #         baseline = self.get_baseline(DIR)
        #         X = X - baseline

        return X, Y

    # def extract_frames_old(self, DIR):
    #     ''' Extract frames for the given subject/session from capgmyo.'''

    #     # Initialize data container for given session
    #     filenames = os.listdir(DIR)
    #     X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
    #     Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

    #     for gdx, name in enumerate(filenames):
    #         mat = sio.loadmat(os.path.join(DIR, name))
    #         emg = mat['data']
    #         emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
    #         cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
    #         if cur_label == 0: continue # exclude rest
    #         else: cur_label -= 1

    #         # Account for exception case of missing repetitions
    #         labels = mat['gesture'].ravel()
    #         labels_rolled = np.roll(np.copy(labels), 1)
    #         delta = (labels - labels_rolled)
    #         indices = np.where(delta != 0)[0]
    #         if len(indices) > 20:
    #             indices = indices[:20]
    #         reps = len(indices) // 2 # number of repetitions is equivalent to half of the number of changepoints
    #         missing = self.num_repetitions - reps # number of missing repetitions from the protocol

    #         # For each repetition available
    #         for idx in range(0, len(indices), 2):
    #             start, end = indices[idx], indices[idx+1]
    #             center = (start + end) // 2 # get the central index of the given repetition
    #             emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
    #             images = self.get_images(emg_segment)

    #             # Add data extracted from given repetition to our data matrix            
    #             X[cur_label, idx//2, :, :, :, :] = images # add EMG surface images onto our data matrix
    #             Y[cur_label, idx//2, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
            
    #         # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
    #         X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

    #     return X, Y
        

# class CapgmyoDataRMS(EMGData):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # self.durations = np.ones((self.num_sessions, self.num_gestures, self.num_repetitions))*2048

#     def extract_frames(self, DIR):
#         ''' Extract frames for the given subject/session from capgmyo.'''

#         # Initialize data container for given session
#         filenames = os.listdir(DIR)
#         X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
#         Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
#         baseline = self.get_baseline(DIR)

#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(DIR, name))
#             emg = mat['data']
#             emg = emg - emg.mean(axis=0, keepdims=True)
#             emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
#             emg = get_rms_signal(emg, Mrms=self.Mrms)
#             cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
#             if cur_label == 0: continue # exclude rest
#             else: cur_label -= 1

#             # Account for exception case of missing repetitions
#             labels = mat['gesture'].ravel()
#             labels_rolled = np.roll(np.copy(labels), 1)
#             delta = (labels - labels_rolled)
#             indices = np.where(delta != 0)[0]
#             if len(indices) > 20:
#                 indices = indices[:20]
#             reps = len(indices) // 2 # number of repetitions is equivalent to half of the number of changepoints
#             missing = self.num_repetitions - reps # number of missing repetitions from the protocol

#             # For each repetition available
#             for idx in range(0, len(indices), 2):
#                 start, end = indices[idx], indices[idx+1]
#                 center = (start + end) // 2 # get the central index of the given repetition                
#                 emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
#                 images = self.get_images(emg_segment)

#                 # Add data extracted from given repetition to our data matrix            
#                 X[cur_label, idx//2, :, :, :, :] = images # add EMG surface images onto our data matrix
#                 Y[cur_label, idx//2, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
            

#             # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
#             X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        
#         # Remove baseline activity
#         if self.remove_baseline:
#             X = X - baseline

#         return X, Y
    

# class CapgmyoDataSegmentRMS(EMGSegmentData):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         # Preinitialize Data tensors
#         self.num_samples = 6349 # largest number of samples found in the dataset for a given movement
#         self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
#         self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

#         # Mask that determines which EMG segments are active
#         self.active = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples), dtype=np.bool_)
#         self.durations = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions)) # durations of gesture segments
    
#     def extract_frames(self, DIR):
#         ''' Extract frames for the given subject/session for CSL dataset.'''

#         # Initialize data container for given session
#         filenames = os.listdir(DIR)
#         X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
#         Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
#         baseline = self.get_baseline(DIR) # get baseline for this given session

#         # Get EMG activity
#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(DIR, name))
#             emg = mat['data']
#             emg = emg - emg.mean(axis=0, keepdims=True)
#             emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)

#             rms = get_rms_signal(emg, Mrms=self.Mrms)
#             cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
#             if cur_label == 0: continue # exclude rest
#             else: cur_label -= 1

#             # Account for exception case of missing repetitions
#             labels = mat['gesture'].ravel()
#             labels_rolled = np.roll(np.copy(labels), 1)
#             delta = (labels - labels_rolled)
#             indices = np.where(delta != 0)[0]
#             if len(indices) > 20:
#                 indices = indices[:20]
#             reps = len(indices) // 2 # number of repetitions is equivalent to half of the number of changepoints
#             missing = self.num_repetitions - reps # number of missing repetitions from the protocol

#             # For each repetition available
#             for idx in range(0, len(indices), 2):
#                 start, end = indices[idx], indices[idx+1] # start and end of provided label
#                 # Get segmentation outcome
#                 emg_segment = emg[start:end]
#                 sdx = int(DIR[-1]) - 1 # get the session number
#                 active_start, active_end = self.is_segment(emg_segment, baseline)
#                 self.active[sdx, cur_label, idx//2, active_start:active_end] = True # set signals to active within that timeframe
#                 self.durations[sdx, cur_label, idx//2] = active_end - active_start # store segment duration in samples

#                 # Add data extracted from given repetition to our data matrix
#                 rms_segment = rms[start:end, :]
#                 images = self.get_images(rms_segment)

#                 X[cur_label, idx//2, active_start:active_end, :, :, :] = images[active_start:active_end] # add EMG surface images onto our data matrix
#                 Y[cur_label, idx//2, active_start:active_end] = np.array([cur_label]*(active_end-active_start))  # add labels onto our label matrix

#             # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
#             X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

#         # Remove baseline activity
#         if self.remove_baseline:
#             X = X - baseline

#         return X, Y
    

############################################################## CSL EMG TENSORIZERS #####################################################################


class CSLData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.is_segment: # if applying activity segmentation
            # Preinitialize Data tensors
            self.num_samples = 3*self.fs
            self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
            self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

                    # Mask that determines which EMG segments are active
            self.active = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples), dtype=np.bool_)
            self.durations = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions)) # durations of gesture segments

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        SESSION_DIR = os.path.join(DIR, self.sub, f"session{self.current_session+1}")
        filenames = os.listdir(SESSION_DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        # Filter out gestures not in subset
        filenames = [name for name in filenames if (int(name.replace('gest', '').replace('.mat', ''))-1) in self.gest_subset]
        filenames = sorted(filenames, key=lambda x: int(x.replace('gest', '').replace('.mat', ''))) # ensures ordering of gestures
        
        # Estimate baseline activity for later removal
        if self.remove_baseline or self.is_segment:
            baseline = self.get_baseline(SESSION_DIR)

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(SESSION_DIR, name))
            # cur_label = int(name.replace('gest', '').replace('.mat', '')) - 1 # get the label for the given gesture

            # Account for exception case of missing repetitions
            reps = mat['gestures'].shape[0]
            missing = self.num_repetitions - reps # number of missing repetitions from the protocol

            # For each repetition available 
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = emg - emg.mean(axis=0, keepdims=True) # centering each channel of EMG to remove baseline drifts
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)

                # Get segmentation outcome
                if self.is_segment:
                    start, end = self.segment(emg, baseline)
                    self.active[self.current_session, gdx, idx, start:end] = True # set signals to active within that timeframe
                    self.durations[self.current_session, gdx, idx] = end - start # store segment duration in samples
                else:
                    center = len(emg) // 2 # get the central index of the given repetition
                    emg = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                
                if self.rms:
                    emg = get_rms_signal(emg, Mrms=self.Mrms)
                    if self.remove_baseline:
                        if self.remove_baseline == 'root-mean-square':
                            emg = emg - baseline
                        elif self.remove_baseline == 'mean-square':
                            emg_squared = emg**2 # get MS from RMS
                            emg_squared = emg_squared - baseline # remove baseline activity from the images
                            emg_squared[emg_squared < 0] = 0 # ensure all values are positive
                            emg = np.sqrt(emg_squared) # get RMS from MS

                images = self.get_images(emg)

                # Add data extracted from given repetition to our data matrix            
                X[gdx, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[gdx, idx, :] = np.array([gdx]*self.num_samples)  # add labels onto our label matrix
        
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, gdx, reps, missing)

        return X, Y
    

# class CSLDataRMS(EMGData):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def extract_frames(self, DIR):
#         ''' Extract frames for the given subject/session for CSL dataset.'''

#         # Initialize data container for given session
#         filenames = os.listdir(DIR)
#         X = np.zeros((self.num_gestures, self.num_repetitions, self.fs, 1, self.input_shape[0], self.input_shape[1]))
#         Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
#         baseline = self.get_baseline(DIR)

#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(DIR, name))
#             cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
#             if cur_label == 0: continue
#             else: cur_label -= 1

#             # Account for exception case of missing repetitions
#             reps = mat['gestures'].shape[0]
#             missing = self.num_repetitions - reps # number of missing repetitions from the protocol

#             # For each repetition available 
#             for idx in range(reps):
#                 emg = mat['gestures'][idx, 0].T
#                 emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
#                 # emg = bandpass(emg, fs=self.fs)
#                 emg = get_rms_signal(emg, Mrms=self.Mrms)
#                 center = len(emg) // 2 # get the central index of the given repetition
#                 emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
#                 images = self.get_images(emg_segment)

#                 # Add data extracted from given repetition to our data matrix            
#                 X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
#                 Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix

#             # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
#             X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        
#         # Remove baseline activity
#         if self.remove_baseline:
#             X = X - baseline

#         return X, Y


# class CSLDataSegmentRMS(EMGSegmentData):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs) 

#         # Preinitialize Data tensors
#         self.num_samples = 3*self.fs
#         self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
#         self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

#         # Mask that determines which EMG segments are active
#         self.active = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples), dtype=np.bool_)
#         self.durations = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions)) # durations of gesture segments

#     def extract_frames(self, DIR):
#         ''' Extract frames for the given subject/session for CSL dataset.'''

#         # Initialize data container for given session
#         filenames = os.listdir(DIR)
#         X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
#         Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
#         baseline = self.get_baseline(DIR)
        
#         # Extract gestures with segmentation algorithm
#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(DIR, name))
#             cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
#             if cur_label == 0: continue
#             else: cur_label -= 1

#             # Account for exception case of missing repetitions
#             reps = mat['gestures'].shape[0]
#             missing = self.num_repetitions - reps # number of missing repetitions from the protocol

#             # For each repetition available 
#             for idx in range(reps):
#                 emg = mat['gestures'][idx, 0].T
#                 emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)

#                 # Get segmentation outcome
#                 start, end = self.segment(emg, baseline)
#                 self.active[self.current_session, cur_label, idx, start:end] = True # set signals to active within that timeframe
#                 self.durations[self.current_session, cur_label, idx] = end - start # store segment duration in samples

#                 emg = get_rms_signal(emg, Mrms=self.Mrms)
#                 images = self.get_images(emg)

#                 # Add data extracted from given repetition to our data matrix            
#                 X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
#                 Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
        
#             # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
#             X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        
#         # Remove baseline activity
#         if self.remove_baseline:
#             X = X - baseline

#         return X, Y


############################################################## HYSER EMG TENSORIZERS #####################################################################


class HyserData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = 2*self.fs # 2 seconds central to the repetition

        # Preinitialize Data tensors
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        sub_int = int(self.sub.replace('subject',''))
        sub = f"0{sub_int}" if sub_int < 10 else str(sub_int)
        curdir = f"subject{sub}_session{self.current_session+1}"
        SESSION_DIR = os.path.join(DIR, curdir)
        filenames = os.listdir(SESSION_DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        # Get baseline from MVC dataset
        if self.remove_baseline:
            baseline = self.get_baseline(SESSION_DIR)

        # baseline = np.zeros((1, 1, 1, 1, self.input_shape[0], self.input_shape[1]))
        # baseline_samp_count = 0
        # Load labels for each recording
        with open(os.path.join(SESSION_DIR, 'label_maintenance.txt'), 'r') as f:
            labels_txt = f.read().split(',')

        labels = {idx+1: int(lab) for idx, lab in enumerate(labels_txt)} # get labels for each text file
        gest_count = 0
        for gdx in range(len(set(labels_txt))):
        # for gdx, gest in enumerate(self.gest_subset):
            label_keys = [key for key in labels.keys() if labels[key] == gdx+1]
            for rep_idx, label_key in enumerate(label_keys): # for each of the two trials per gesture
                record = wfdb.rdrecord(os.path.join(SESSION_DIR, f"maintenance_raw_sample{label_key}"))
                emg = record.p_signal
                emg = emg - emg.mean(axis=0, keepdims=True) # centering each channel of EMG to remove baseline drifts
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                if self.rms:
                    emg = get_rms_signal(emg, Mrms=self.Mrms)

                    if self.remove_baseline:
                        if self.remove_baseline == 'root-mean-square':
                            emg = emg - baseline
                        elif self.remove_baseline == 'mean-square':
                            emg_square = emg ** 2
                            emg_square = emg_square - baseline
                            emg_square[emg_square < 0] = 0
                            emg = np.sqrt(emg_square)

                if gdx in self.gest_subset:
                    # Keep only central 2s
                    center = emg.shape[0]//2
                    emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                    images = self.get_images(emg_segment) 
                    
                    # Add data extracted from given repetition to our data matrix            
                    X[floor(gest_count), rep_idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                    Y[floor(gest_count), rep_idx, :] = np.array([floor(gest_count)]*self.num_samples)  # add labels onto our label matrix
                    gest_count += 1/len(label_keys)

        return X, Y
    
    
############################################################## GRABMYO EMG TENSORIZERS #####################################################################


class GrabmyoData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = 2*self.fs # 2 second central to the repetition

        # Preinitialize Data tensors
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        sub_int = int(self.sub.replace('subject',''))
        subdir = f"session{self.current_session+1}_participant{sub_int}"
        session_sub_dir = os.path.join(f"Session{self.current_session+1}", subdir)
        SESSION_DIR = os.path.join(DIR, session_sub_dir)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        # Estimate baseline activity for later removal
        if self.remove_baseline or self.is_segment:
            baseline = self.get_baseline(SESSION_DIR)

        for gdx, gest in enumerate(self.gest_subset):
            for rep_idx in range(self.num_repetitions): # 7 repetitions per gesture
                rec_name = f"{subdir}_gesture{gest+1}_trial{rep_idx+1}"
                record = wfdb.rdrecord(os.path.join(SESSION_DIR, rec_name))
                emg = record.p_signal
                keep_channels = ['U' not in name for name in record.sig_name] # drop redundant channels
                emg = emg[:, keep_channels]
                emg = emg - emg.mean(axis=0, keepdims=True)
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                if self.rms:
                    emg = get_rms_signal(emg, Mrms=self.Mrms)
                    if self.remove_baseline:
                        if self.remove_baseline == 'root-mean-square':
                            emg = emg - baseline
                        elif self.remove_baseline == 'mean-square':
                            emg_square = emg ** 2
                            emg_square = emg_square - baseline
                            emg_square[emg_square < 0] = 0
                            emg = np.sqrt(emg_square)
                
                # Keep only central 2s
                center = emg.shape[0]//2
                emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = self.get_images(emg_segment) 
                            
                # Add data extracted from given repetition to our data matrix            
                X[gdx, rep_idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[gdx, rep_idx, :] = np.array([gdx]*self.num_samples)  # add labels onto our label matrix

        return X, Y