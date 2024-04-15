import os
from time import time
from copy import deepcopy

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import torch

from utils.emg_processing import bandpass, bandstop, identity, get_rms_signal


class EMGData:
    
    def __init__(self, path='../datasets/csl', sub='subject1', transform=None, target_transform=None, norm=0,
                  num_gestures=8, num_repetitions=10, input_shape=(8, 16), fs=1000, sessions='session1', intrasession=False):
        # Store all appropriate data parameters
        self.path = path
        self.fs = fs
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

        # Preinitialize Data tensors
        self.X = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        self.Y = np.zeros((self.num_sessions, self.num_gestures, self.num_repetitions, self.num_samples))

        # Target transforms
        self.transform = transform
        self.target_transform = target_transform

    def extract_frames(self, DIR=None):
        ''' Placeholder function to be overriden by child classes.'''
        X = np.zeros((self.num_gestures, self.num_repetitions, self.fs, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))
        return X, Y

    def load_tensors(self):
        ''' Takes in data files and cretes complete data tensor for either intrasession or intersession case.'''
        intrasession = not isinstance(self.sessions, list) # if a session selected, only load that given session for intrasession performance

        for idx, session in enumerate(self.sessions):
            self.current_session = idx+1
            DIR = os.path.join(self.path, self.sub, session)
            Xs, Ys = self.extract_frames(DIR)
            self.X[idx, :, :, :, :, :, :] = Xs # add data extracted from given session
            self.Y[idx, :, :, :] = Ys # add labels extracted from given session

        # Convert data to tensor
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)
    
    def get_tensors(self, test_idx):
        ''' Return data in desired format of surface images, with a leave-one-out approach for testing.
        '''
        # Apply normalization prior to dataloader
        self.normalize_images()

        if self.intrasession:
            idxs = list(range(self.num_repetitions))
            test_idx = idxs.pop(test_idx)
            X_train = torch.flatten(self.X[:, :, idxs, :, :, :, :], end_dim=3) # get all but one repetition
            Y_train = torch.flatten(self.Y[:, :, idxs, :], end_dim=3) # get all but one repetition
            X_test = torch.flatten(self.X[:, :, [test_idx], :, :, :, :], end_dim=3) # get one repetition
            Y_test = torch.flatten(self.Y[:, :, [test_idx], :],end_dim=3) # get one repetition
        else:
            idxs = list(range(self.num_sessions))
            test_idx = idxs.pop(test_idx)
            X_train = torch.flatten(self.X[idxs, :, :, :, :, :, :], end_dim=3) # get all sessions but one
            Y_train = torch.flatten(self.Y[idxs, :, :, :], end_dim=3) # get all sessions but one
            X_test = torch.flatten(self.X[[test_idx], :, :, :, :, :, :], end_dim=3) # get other session
            Y_test = torch.flatten(self.Y[[test_idx], :, :, :], end_dim=3) # get other session

        # Convert to torch tensors
        X_train, X_test = X_train.to(torch.float32), X_test.to(torch.float32)

        return X_train, Y_train, X_test, Y_test
    
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

    def normalize_images(self):
        '''Placeholder to be overridden by child classes'''
        pass


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
            # cur_label = int(name[4:7].lstrip('0')) - 1 # get the label for the given gesture

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
                images = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = images.reshape(self.num_samples, 1, self.input_shape[0], self.input_shape[1], order='F')

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx//2, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx//2, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
            
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

        return X, Y

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
                images = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = np.flip(images.reshape(self.num_samples, 1, 8, 24, order='F'), axis=0)
                images = images[:, :, 1:, :] # drop first row given bipolar nature of data and create list

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
        
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

            return X, Y
        

class CapgmyoDataRMS(EMGData):
    
    def __init__(self, M=150, median_filt=False, **kwargs):
        super().__init__(**kwargs)
        self.median_filt = median_filt
        self.M = M         

    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session from capgmyo.'''

        # Initialize data container for given session
        filenames = os.listdir(DIR)
        X = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples, 1, self.input_shape[0], self.input_shape[1]))
        Y = np.zeros((self.num_gestures, self.num_repetitions, self.num_samples))

        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            emg = mat['data']
            emg = emg - emg.mean(axis=0, keepdims=True)
            emg = bandstop(bandpass(emg, fs=self.fs), fs=self.fs)
            emg = get_rms_signal(emg, M=self.M)
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
                images = emg[center - self.num_samples//2 : center + self.num_samples//2, :]
                images = images.reshape(self.num_samples, 1, self.input_shape[0], self.input_shape[1], order='F')

                # Add data extracted from given repetition to our data matrix            
                X[cur_label, idx//2, :, :, :, :] = images # add EMG surface images onto our data matrix
                Y[cur_label, idx//2, :] = np.array([cur_label]*self.num_samples)  # add labels onto our label matrix
            
            # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
            X, Y = self.oversample_repetitions(X, Y, cur_label, reps, missing)

        return X, Y

    def uniform_grid(self, images, dg=27.5):
        ''' Takes in given images and interpolates as to estimate evenly spaced electrodes.'''

        # Get initial coordinates from capgmyo and symmetric grid
        IED = 10
        dv, dh = 8.5, 8.0
        H, W = 8*dv, 7*dg + 8*dh
        self.Nv, self.Nh = int(H // IED) + 1, int(W // IED) + 1
        x, y, xnew, ynew = self.capgmyo_coords(dg)
        yv, xv = np.meshgrid(ynew, xnew, indexing='ij')
        points = np.vstack([yv.reshape(-1, order='F'), xv.reshape(-1, order='F')]) # where to sample points for each grid system
        images_reg = np.zeros((images.shape[0], 1, self.Nv, self.Nh))

        # For every surface image, apply interpolation
        for n in range(images.shape[0]):
            cpg2reg = RegularGridInterpolator((y,x), images[n,0,:,:], bounds_error=False, fill_value=0.0, method='linear')
            images_reg[n,0,:,:] = cpg2reg(points.T).reshape(self.Nv, self.Nh, order='F')
        return images_reg

    def capgmyo_coords(self, dg):
        ''' Get the coordinates of Capgmyo, given distance between grids (dg), normalized
            by circumference/grid height in respective dimensions.
        '''
        dh, dv = 8.0, 8.5 # grid electrode distances
        W, H = 8*(dh + dg) - dg, 8*dv # total surface width and height

        # Horizontal coordinates
        Nh = W // 10 + 1 # How many electrodes we can get evenly spaced with 1mm spacing
        init_gap_h = (W % 10)/2 # divided by two since we want same spacing between beginning and end

        x = np.zeros(16)
        w = 0
        for idx in range(16):
            x[idx] = w
            if (idx % 2) == 0:
                w += dh
            else:
                w += dg

        # Vertical coordinates
        Nv = H // 10 + 1
        init_gap_v = (H % 10)/2
        y = np.linspace(0, H, num=8)*dv

        # Get new coordinates
        xnew = np.arange(Nh)*10 + init_gap_h
        ynew = np.arange(Nv)*10 + init_gap_v
        return x, y, xnew, ynew
    
    def get_tensors(self, test_idx, dg=27.5):
        ''' Keeps parent method, but adds intermediary interpolation step.
        '''
        # Original parent class method
        X_train, Y_train, X_test, Y_test = super().get_tensors(test_idx)

        # 2D regridding!
        print('2D REGRIDDING EMG IMAGE...')
        X_train, X_test = X_train.numpy(), X_test.numpy()
        X_train = self.uniform_grid(X_train, dg=dg)
        X_test = self.uniform_grid(X_test, dg=dg)

        X_train, X_test = torch.tensor(X_train).to(torch.float32), torch.tensor(X_test).to(torch.float32)

        return X_train, Y_train, X_test, Y_test


# class CSLFrameLoader(Dataset):
#     def __init__(self, path='../datasets/csl', sub='subject1', transform=None, target_transform=None,
#                   norm=0, num_gestures=8, num_repetitions=10, intrasession=False, session=False, test_rep=None):
#         super(CSLFrameLoader, self).__init__()

#         self.num_gestures = num_gestures
#         self.num_repetitions = num_repetitions - int(intrasession)
#         self.fs = 2048
#         self.n_samples = self.fs
#         self.channels = 168

#         self.path = os.path.join(path, sub)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.intrasession = intrasession
#         self.test_rep = test_rep

#         # Initialize variables
#         self.X_test, self.Y_test = [], []
#         self.X, self.Y = [], []

#         # If intrasession, store information for only a given session
#         if intrasession:
#             DIR = os.path.join(path, sub, session)
#             X, Y = self.extract_frames(DIR)
#             self.X.extend(X)
#             self.Y.extend(Y)

#         else:
#             sessions = ['session{}'.format(idx) for idx in range(1, 6)]
#             for idx, session in enumerate(sessions):
#                 DIR = os.path.join(path, sub, session)
#                 X, Y = self.extract_frames(DIR)
#                 if test_rep == (idx+1):
#                     self.X_test.extend(X)
#                     self.Y_test.extend(Y)
#                 else:
#                     self.X.extend(X)
#                     self.Y.extend(Y)

#         # Convert data to tensor
#         self.X = torch.tensor(np.array(self.X)).to(torch.float32)
#         self.Y = torch.tensor(np.array(self.Y)).to(torch.float32)
#         if intrasession:
#             self.X_test = torch.tensor(np.array(self.X_test)).to(torch.float32)
#             self.Y_test = torch.tensor(np.array(self.Y_test)).to(torch.float32)

#         if norm  == 1: # standardization
#             self.mean = self.X.mean()
#             self.std = self.X.std()
#             self.X = (self.X - self.mean)/(self.std + 1.e-12)
#             if intrasession: self.X_test = (self.X_test - self.mean)/(self.std + 1.e-12)

#         elif norm == -1: # scale between [-1, 1]
#             self.max = self.X.amax(keepdim=True)
#             self.min = self.X.amin(keepdim=True)
#             self.X = (self.X - self.min)/(self.max - self.min)*2 - 1
#             if intrasession: self.X_test = (self.X_test - self.min)/(self.max - self.min)*2 - 1

#         if transform:
#             self.X = transform(self.X)

#         self.len = self.n_samples * self.num_repetitions * self.num_gestures

#     def __len__(self):
#         return self.len
    
#     def extract_frames(self, DIR):
#         ''' Extract frames for the given subject/session.'''
#         filenames = os.listdir(DIR)
#         X, Y = [], []
#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(DIR, name))
#             reps = mat['gestures'].shape[0] # number of repetitions stored
#             cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
#             if cur_label == 0: continue # exclude rest
#             else: cur_label -= 1 

#             # For each repetition
#             for idx in range(reps):
#                 emg = mat['gestures'][idx, 0].T
#                 emg = bandpass(emg, self.fs)
#                 emg = bandstop(emg, self.fs) 
#                 center = len(emg) // 2 # get the central index of the given repetition
#                 images = emg[center - self.fs//2 : center + self.fs//2, :]
#                 images = np.flip(images.reshape(self.n_samples, 1, 8, 24, order='F'), axis=0)
#                 images = list(images[:, :, 1:, :]) # drop first row given bipolar nature of data and create list
                
#                 # If intrasession, use given repetition for testing
#                 if self.intrasession and (idx+1)==self.test_rep:
#                     self.X_test.extend(images)
#                     self.Y_test.extend( [cur_label]*self.n_samples )
#                 else:
#                     X.extend(images) # add EMG surface images onto our data matrix
#                     Y.extend( [cur_label]*self.n_samples ) # add labels onto our label matrix
#         return X, Y
    
#     def train_test_split(self):
#         '''Gets a deepcopy of the current loader with test data set as training data.'''
#         test_loader = deepcopy(self) # make a copy of current loader
#         test_loader.X, test_loader.Y = deepcopy(test_loader.X_test), deepcopy(test_loader.Y_test)
#         del test_loader.X_test, test_loader.Y_test, self.X_test, self.Y_test
#         test_loader.num_repetitions = 1 # only a single repetition in the given test set
#         if self.intrasession:
#             test_loader.len = len(self) // self.num_repetitions
#         else:
#             test_loader.len = self.n_samples * self.num_repetitions * self.num_gestures
#         return test_loader # returns test loader

#     def __getitem__(self, idx):
#         X, Y = self.X[idx], self.Y[idx]
#         return X, Y
