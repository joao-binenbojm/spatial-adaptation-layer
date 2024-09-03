import os
from time import time
from copy import deepcopy

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
import torch

from emg_processing import bandpass, bandstop, identity, get_rms_signal
from networks_utils import median_pool_2d


class EMGData:
    
    def __init__(self, dataset='csl', path='../datasets/capgmyo/dbb_csl', sub='subject1', transform=None, target_transform=None, norm=0,
                  num_gestures=26, num_repetitions=10, input_shape=(8, 24), fs=2048, Trms=150, sessions='session1', intrasession=False, remove_baseline=False):
        # Store all appropriate data parameters
        self.dataset = dataset
        self.path = path
        self.fs = fs
        self.Mrms = int(self.fs*(Trms / 1000))        
        self.intrasession = intrasession
        self.num_samples = fs
        self.norm = norm
        self.num_gestures = num_gestures
        self.num_repetitions = num_repetitions
        self.input_shape = input_shape
        self.sessions = sessions
        self.num_sessions = len(sessions)
        self.sub = sub
        self.current_session = 0 # to keep track of what session we are extracting from
        self.remove_baseline=False

        # Preinitialize Data tensors
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

        # Target transforms
        self.transform = transform
        self.target_transform = target_transform

    def get_images(self, emg_segment):
        ''' Takes in either raw sEMG or RMS activity and returns it in the appropriate shape given the dataset being used.'''
        if self.dataset == 'csl':
            images = np.flip(np.array(emg_segment).reshape(emg_segment.shape[0], 1, 8, 24, order='F'), axis=0)
            images = images[:, :, 1:, :] # drop first row given bipolar nature of data and create list
        elif self.dataset == 'capgmyo':
            images = np.array(emg_segment).reshape(emg_segment.shape[0], 1, self.input_shape[0], self.input_shape[1], order='F')
        else:
            raise Exception("No dataset specified.")
        return images 
        
    def get_baseline(self, DIR):
        '''Given the subject/session directory, computes the baseline activity for every channel, depending on the given dataset.'''
        if self.dataset == 'csl':
            mat = sio.loadmat(os.path.join(DIR, 'gest0.mat'))
            reps = mat['gestures'].shape[0]
            baseline = np.zeros((1,1,1,1,7,24))
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                emg = get_rms_signal(emg, Mrms=self.Mrms)
                images = self.get_images(emg)
                images = images.reshape(1, 1, *images.shape)
                baseline += images.mean(axis=2, keepdims=True)
            baseline = baseline/reps

        elif self.dataset == 'capgmyo':
            filenames = os.listdir(DIR)
            baseline = np.zeros((1, 1, 1, 1, self.input_shape[0], self.input_shape[1]))
            ngests = len(filenames)
            for gdx, name in enumerate(filenames):
                mat = sio.loadmat(os.path.join(DIR, name))
                emg, labels = mat['data'], mat['gesture'].ravel()
                emg = emg - emg.mean(axis=0, keepdims=True)
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                rms = get_rms_signal(emg, Mrms=self.Mrms)
                images = self.get_images(rms)
                images = images.reshape(1, 1, *images.shape)
                baseline += images[:,:, labels==0, :, :, :].mean(axis=2, keepdims=True)
            baseline / ngests

        else:
            raise Exception("No dataset specified.")
        return baseline

    def extract_frames(self, DIR=None):
        ''' Placeholder function to be overriden by child classes.'''
        X = np.zeros((self.num_gestures, self.num_repetitions, self.fs, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        return X, Y

    def load_tensors(self):
        ''' Takes in data files and cretes complete data tensor for either intrasession or intersession case.'''
        for idx, session in enumerate(self.sessions):
            DIR = os.path.join(self.path, self.sub, session)
            Xs, Ys = self.extract_frames(DIR)
            self.X[idx, :, :, :, :, :, :] = Xs # add data extracted from given session
            self.Y[idx, :, :, :] = Ys # add labels extracted from given session
            self.current_session += 1

        # Convert data to tensor
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)
    
    def get_tensors(self, train_session=None, test_session=None, rep_idx=None, gest_idxs=None):
        ''' Return data in desired format of surface images, with a leave-one-out approach for testing.
        '''
        if self.intrasession:
            idxs = list(range(self.num_repetitions))
            test_idx = idxs.pop(rep_idx)
            X_train = torch.flatten(self.X[[test_session], :, idxs, :, :, :, :], end_dim=-4) # get all but one repetition
            Y_train = torch.flatten(self.Y[[test_session], :, idxs, :], end_dim=-1) # get all but one repetition
            X_test = torch.flatten(self.X[[test_session], :, [test_idx], :, :, :, :], end_dim=-4) # get one repetition
            Y_test = torch.flatten(self.Y[[test_session], :, [test_idx], :], end_dim=-1) # get one repetition
            
            # Return duration of each segment in test set, which are all 1s unless specified
            test_durations = self.num_samples*np.ones(self.Y.shape[1])

            # Convert to torch tensors of type float32
            X_train, X_test = X_train.to(torch.float32), X_test.to(torch.float32)
            return X_train, Y_train, X_test, Y_test, test_durations
        
        else:
            idxs = list(range(self.num_repetitions))
            # If fine-tuning on a single repetition
            if rep_idx is not None:
                adapt_idx = [idxs.pop(rep_idx)]
            else: # else, fine-tune on all available test data
                adapt_idx = idxs

            # If fine-tuning on a single repetition of one or few gestures
            if gest_idxs is not None:
                if isinstance(gest_idxs, int): gest_idxs = [gest_idxs] # if single gesture for calibration selected
            else:
                gest_idxs = np.arange(self.num_gestures)

            X_train = torch.flatten(self.X[[train_session], :, :, :, :, :, :], end_dim=-4) # get train session labels
            Y_train = torch.flatten(self.Y[[train_session], :, :, :], end_dim=-1) # get train session labels
            X_adapt = torch.flatten(self.X[[test_session], gest_idxs, adapt_idx, :, :, :, :], end_dim=-4) # get all sessions but one
            Y_adapt = torch.flatten(self.Y[[test_session], gest_idxs, adapt_idx, :], end_dim=-1) # get train session labels
            X_test = torch.flatten(self.X[[test_session], :, idxs, :, :, :, :], end_dim=-4) # get other session
            Y_test = torch.flatten(self.Y[[test_session], :, idxs, :], end_dim=-1) # get other session

             # Return duration of each segment in test set, which are all 1s unless specified
            test_durations = self.num_samples*np.ones(self.Y.shape[1]*(self.Y.shape[2] - 1))

            # Convert to torch tensors of type float32
            X_train, X_adapt, X_test = X_train.to(torch.float32), X_adapt.to(torch.float32), X_test.to(torch.float32)
            return X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations
    
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


class EMGSegmentData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        images = self.get_images(rms)
        baseline = baseline[0,0,:,:,:,:] # remove first two singleton dimensions for easier baseline subtraction
        bs_imgs = images - baseline # remove baseline activity from the images
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

    def get_tensors(self, train_session=None, test_session=None, rep_idx=None, gest_idxs=None):
        ''' Return data in desired format of surface images, with a leave-one-out approach for testing.
        '''
        if self.intrasession:
            idxs = list(range(self.num_repetitions))
            test_idx = idxs.pop(rep_idx)

            # Get appropriate train/test/adapt split
            X_train = self.X[[test_session], :, idxs, :, :, :, :]
            Y_train = self.Y[[test_session], :, idxs, :]
            train_active = self.active[[test_session], :, idxs, :]
            X_test = self.X[[test_session], :, [test_idx], :, :, :, :]
            Y_test = self.Y[[test_session], :, [test_idx], :]
            test_active = self.active[[test_session], :, [test_idx], :]

            # Get only detected active segments of activity
            X_train, Y_train = X_train[torch.tensor(train_active)], Y_train[torch.tensor(train_active)]
            X_test, Y_test = X_test[torch.tensor(test_active)], Y_test[torch.tensor(test_active)]
            test_durations = self.durations[test_session, :, test_idx]         
            
            # Convert to torch tensors of type float32
            X_train, X_test = X_train.to(torch.float32), X_test.to(torch.float32)
            return X_train, Y_train, X_test, Y_test, test_durations
        
        else:
            idxs = list(range(self.num_repetitions))
            # If fine-tuning on a single repetition
            if rep_idx is not None:
                adapt_idx = [idxs.pop(rep_idx)]
            else: # else, fine-tune on all available test data
                adapt_idx = idxs

            # Get appropriate train/test/adapt split
            X_train = self.X[[train_session], :, :, :, :, :, :]
            Y_train = self.Y[[train_session], :, :, :]
            train_active = self.active[[train_session], :, :, :]
            X_adapt = self.X[[test_session], :, adapt_idx, :, :, :, :]
            Y_adapt = self.Y[[test_session], :, adapt_idx, :]
            adapt_active = self.active[[test_session], :, adapt_idx, :]
            X_test = self.X[[test_session], :, idxs, :, :, :, :]
            Y_test = self.Y[[test_session], :, idxs, :]
            test_active = self.active[[test_session], :, idxs, :]

            # If fine-tuning on a single repetition of one or few gestures
            if gest_idxs is not None:
                if isinstance(gest_idxs, int): gest_idxs = [gest_idxs] # if single gesture for calibration selected
                X_adapt = X_adapt[:, gest_idxs, :, :, :, :] # only keep the gestures for calibration
                Y_adapt = Y_adapt[:, gest_idxs, :]
                adapt_active = adapt_active[:, gest_idxs, :]

            # Get only detected active segments of activity
            X_train, X_adapt, X_test = X_train[torch.tensor(train_active)], X_adapt[torch.tensor(adapt_active)], X_test[torch.tensor(test_active)]
            Y_train, Y_adapt, Y_test = Y_train[torch.tensor(train_active)], Y_adapt[torch.tensor(adapt_active)], Y_test[torch.tensor(test_active)]
            test_durations = self.durations[test_session, :, idxs]

            # Convert to torch tensors of type float32
            X_train, X_adapt, X_test = X_train.to(torch.float32), X_adapt.to(torch.float32), X_test.to(torch.float32)
            return X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations.ravel()

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
            dur_sample = self.durations[self.current_session, cur_label, rep_idx]
            self.durations[self.current_session, cur_label, idx] = dur_sample
            active_sample = self.active[self.current_session, cur_label, rep_idx, :]
            self.active[self.current_session, cur_label, idx, :] = active_sample

        return X, Y

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

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            emg = mat['data']
            emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue # exclude rest
            else: cur_label -= 1

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

        return X, Y
        

class CapgmyoDataRMS(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.durations = np.ones((self.num_sessions, self.num_gestures, self.num_repetitions))*2048

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session from capgmyo.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        baseline = self.get_baseline(DIR)

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            emg = mat['data']
            emg = emg - emg.mean(axis=0, keepdims=True)
            emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
            emg = get_rms_signal(emg, Mrms=self.Mrms)
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue # exclude rest
            else: cur_label -= 1

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

            ## TESTING ##
            if self.durations[self.current_session, cur_label].min() == 0:
                print('WAIT')
        
        # Remove baseline activity
        if self.remove_baseline:
            X = X - baseline

        return X, Y
    

class CapgmyoDataSegmentRMS(EMGSegmentData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Preinitialize Data tensors
        self.num_samples = 6349 # largest number of samples found in the dataset for a given movement
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

        # Mask that determines which EMG segments are active
        self.active = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples), dtype=np.bool_)
        self.durations = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions)) # durations of gesture segments
    
    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        baseline = self.get_baseline(DIR) # get baseline for this given session

        # Get EMG activity
        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            emg = mat['data']
            emg = emg - emg.mean(axis=0, keepdims=True)
            emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)

            rms = get_rms_signal(emg, Mrms=self.Mrms)
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue # exclude rest
            else: cur_label -= 1

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
                start, end = indices[idx], indices[idx+1] # start and end of provided label
                # Get segmentation outcome
                emg_segment = emg[start:end]
                sdx = int(DIR[-1]) - 1 # get the session number
                active_start, active_end = self.segment(emg_segment, baseline)
                self.active[sdx, cur_label, idx//2, active_start:active_end] = True # set signals to active within that timeframe
                self.durations[sdx, cur_label, idx//2] = active_end - active_start # store segment duration in samples

                # Add data extracted from given repetition to our data matrix
                rms_segment = rms[start:end, :]
                images = self.get_images(rms_segment)

                X[cur_label, idx//2, active_start:active_end, :, :, :] = images[active_start:active_end] # add EMG surface images onto our data matrix
                Y[cur_label, idx//2, active_start:active_end] = np.array([cur_label]*(active_end-active_start))  # add labels onto our label matrix

            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

        # Remove baseline activity
        if self.remove_baseline:
            X = X - baseline

        return X, Y
    

############################################################## CSL EMG TENSORIZERS #####################################################################


class CSLData(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.fs, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue # exclude rest
            else: cur_label -= 1

            # Account for exception case of missing repetitions
            reps = mat['gestures'].shape[0]
            missing = self.num_repetitions - reps # number of missing repetitions from the protocol

            # For each repetition available 
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                center = len(emg) // 2 # get the central index of the given repetition
                emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = self.get_images(emg_segment)

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
        
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        return X, Y
    

class CSLDataRMS(EMGData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session for CSL dataset.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.fs, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        baseline = self.get_baseline(DIR)

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue
            else: cur_label -= 1

            # Account for exception case of missing repetitions
            reps = mat['gestures'].shape[0]
            missing = self.num_repetitions - reps # number of missing repetitions from the protocol

            # For each repetition available 
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
                # emg = bandpass(emg, fs=self.fs)
                emg = get_rms_signal(emg, Mrms=self.Mrms)
                center = len(emg) // 2 # get the central index of the given repetition
                emg_segment = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = self.get_images(emg_segment)

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix

            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        
        # Remove baseline activity
        if self.remove_baseline:
            X = X - baseline

        return X, Y


class CSLDataSegmentRMS(EMGSegmentData):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

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
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        baseline = self.get_baseline(DIR)
        
        # Extract gestures with segmentation algorithm
        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue
            else: cur_label -= 1

            # Account for exception case of missing repetitions
            reps = mat['gestures'].shape[0]
            missing = self.num_repetitions - reps # number of missing repetitions from the protocol

            # For each repetition available 
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)

                # Get segmentation outcome
                start, end = self.segment(emg, baseline)
                self.active[self.current_session, cur_label, idx, start:end] = True # set signals to active within that timeframe
                self.durations[self.current_session, cur_label, idx] = end - start # store segment duration in samples

                emg = get_rms_signal(emg, Mrms=self.Mrms)
                images = self.get_images(emg)

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
        
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)
        
        # Remove baseline activity
        if self.remove_baseline:
            X = X - baseline

        return X, Y

