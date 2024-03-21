import os

import scipy.io as sio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import signal

import preprocess_functions as preprocess_functions
from collections import Counter
from copy import deepcopy

from utils_emg import bandpass, bandstop, identity

def extract_frames_csl(DIR, num_gestures=26, num_repetitions=10, fs=2048, filters=identity, input_shape=(7,24)):
    ''' Extract frames for the given subject/session.'''

    # Initialize data container for given session
    num_samples = fs # 1 second interval
    filenames = os.listdir(DIR)
    X = np.zeros((num_gestures, num_repetitions, fs, 1, input_shape[0], input_shape[1]))
    Y = np.zeros((num_gestures, num_repetitions, num_samples))

    for gdx, name in enumerate(filenames):
        mat = sio.loadmat(os.path.join(DIR, name))
        cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
        if cur_label == 0: continue # exclude rest
        else: cur_label -= 1

        # Account for exception case of missing repetitions
        reps = mat['gestures'].shape[0]
        missing = num_repetitions - reps # number of missing repetitions from the protocol

        # For each repetition available 
        for idx in range(reps):
            emg = mat['gestures'][idx, 0].T
            emg = filters(emg, fs=fs)
            center = len(emg) // 2 # get the central index of the given repetition
            images = emg[center - num_samples//2 : center + num_samples//2, :]
            images = np.flip(images.reshape(num_samples, 1, 8, 24, order='F'), axis=0)
            images = images[:, :, 1:, :] # drop first row given bipolar nature of data and create list

            # Add data extracted from given repetition to our data matrix            
            X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
            Y[cur_label, idx, :] = np.array([cur_label]*num_samples)  # add labels onto our label matrix
        
        # For each repetition that is missing from total number of repetitions, oversample a previous repetition
        for idx in range(reps, reps+missing):
            rep_idx = np.random.randint(0, reps)

            Xsample = X[cur_label, rep_idx, :, :, :, :] # get entire repetition
            X[cur_label, idx, :, :, :, :] = Xsample
            Ysample = Y[cur_label, rep_idx, :]
            Y[cur_label, idx, :] = Ysample

    return X, Y

def extract_frames_capgmyo(filenames, num_gestures=8, num_repetitions=10, num_samples=1000, input_shape=(8,16)):
    ''' Extract frames for the given subject/session.'''

    # Initialize data container for given session
    X = np.zeros((num_gestures, num_repetitions, num_samples, 1, input_shape[0], input_shape[1]))
    Y = np.zeros((num_gestures, num_repetitions, num_samples))

    for gdx, name in enumerate(filenames):
        mat = sio.loadmat(name)
        emg = mat['data']
        cur_label = int(name[4:7].lstrip('0')) - 1 # get the label for the given gesture

        # Account for exception case of missing repetitions
        labels = mat['gesture'].ravel()
        labels_rolled = np.roll(np.copy(labels), 1)
        delta = (labels - labels_rolled)
        indices = np.where(delta != 0)[0]
        reps = indices // 2 # number of repetitions is equivalent to half of the number of changepoints
        missing = num_repetitions - reps # number of missing repetitions from the protocol

        # For each repetition available
        for idx in range(0, len(indices), 2):
            start, end = indices[idx], indices[idx+1]
            center = (start + end) // 2 # get the central index of the given repetition
            emg = bandpass(emg)
            # emg = bandstop(emg) 
            images = emg[center - num_samples//2 : center + num_samples//2, :]
            images = images.reshape(num_samples, 1, 8, 16, order='F')

            # Add data extracted from given repetition to our data matrix            
            X[cur_label, idx, :, :, :, :] = images # add EMG surface images onto our data matrix
            Y[cur_label, idx, :] = np.array([cur_label]*num_samples)  # add labels onto our label matrix
        
        # For each repetition that is missing from total number of repetitions, oversample from previous repetitions
        for idx in range(reps, reps+missing):
            rep_idx = np.random.randint(0, reps)

            Xsample = X[cur_label, rep_idx, :, :, :, :]
            X[cur_label, idx, :, :, :, :] = Xsample
            Ysample = Y[cur_label, rep_idx, :]
            Y[cur_label, idx, :] = Ysample

    return np.array(X), np.array(Y)

def load_tensors(path='../datasets/csl', sub='subject1', extract_frames=extract_frames_csl, transform=None, target_transform=None,
                  norm=0, num_gestures=8, num_repetitions=10, input_shape=(8, 16), fs=1000, filters=identity, sessions='session1'):
    ''' Takes in data files and cretes complete data tensor for either intrasession or intersession case.'''
    num_samples = fs # equivalent to 1s worth of samples
    intrasession = not isinstance(sessions, list) # if a session selected, only load that given session for intrasession performance
    if intrasession: num_sessions = 1
    else: num_sessions = len(sessions)

    # Given data containers
    X = np.zeros((num_sessions, num_gestures, num_repetitions, num_samples, 1, input_shape[0], input_shape[1]))
    Y = np.zeros((num_sessions, num_gestures, num_repetitions, num_samples))

    # If intrasession, store information for only a given session
    if intrasession:
        DIR = os.path.join(path, sub, sessions)
        Xs, Ys = extract_frames(DIR, num_gestures=num_gestures, num_repetitions=num_repetitions, fs=fs, filters=filters, input_shape=input_shape)
        X[0, :, :, :, :, :, :], Y[0, :, :, :] = Xs, Ys # set data to frames extracted from single session

    else:
        for idx, session in enumerate(sessions):
            DIR = os.path.join(path, sub, session)
            X, Y = extract_frames(DIR, num_gestures=num_gestures, num_repetitions=num_repetitions, fs=fs, filters=filters, input_shape=input_shape)
            X[idx, :, :, :, :, :, :] = X # add data extracted from given session
            Y[idx, :, :, :] = Y # add labels extracted from given session

    # Convert data to tensor
    X = torch.tensor(X).to(torch.float32)
    Y = torch.tensor(Y).to(torch.float32)
    return X, Y

# Dataset then just very simply takes in images and labels and does preprocessing/train test splits
class EMGFrameLoader(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None, norm=0, train=True, stats=None):
        super(EMGFrameLoader, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.stats = stats
        self.X = X
        self.Y = Y

        if norm  == 1: # standardization
            if train:
                self.mean = self.X.mean()
                self.std = self.X.std()
            else:
                self.mean = stats['mean'] # use training stats 
                self.std = stats['std']
            self.stats = {'mean': self.mean, 'std': self.std}
            self.X = (self.X - self.mean)/(self.std + 1.e-12)

        elif norm == -1: # scale between [-1, 1]
            if train:
                self.max = self.X.amax(keepdim=True)
                self.min = self.X.amin(keepdim=True)
            else:
                self.max = stats['max']
                self.min = stats['min']
            self.stats = {'max': self.max, 'min': self.min}
            self.X = (self.X - self.min)/(self.max - self.min)*2 - 1

        if transform:
            self.X = transform(self.X)

        self.len = X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        return X, Y

# def get_indices(labels):
#     ''' Determines at which indices the labels change (i.e. a new gesture label has begun).'''
#     labels_rolled = np.roll(np.copy(labels), 1)
#     delta = (labels - labels_rolled)
#     indices = np.where(delta != 0)[0]
#     return indices
# class CapgmyoFrameLoader(Dataset):
#     def __init__(self, path='../datasets/capgmyo', db='dba', sub='001', transform=None, target_transform=None,
#                   norm=0, num_gestures=8, num_repetitions=10, intrasession=False, test_rep=None):
#         super(CapgmyoFrameLoader, self).__init__()

#         self.num_gestures = num_gestures
#         self.num_repetitions = num_repetitions - int(intrasession)
#         self.n_samples = 1000
#         self.fs = 1000
#         self.channels = 128

#         self.path = os.path.join(path, db)
#         self.transform = transform
#         self.target_transform = target_transform

#         # Filter for only files of the given subject
#         filenames = os.listdir(self.path)
#         filenames = [name for name in filenames if (name[:3] == sub) and (name[4] == '0')]

#         # Extract EMG 'frames' from capgmyo
#         self.X, self.Y = [], []
#         if intrasession:
#             self.X_test, self.Y_test = [], []

#         # For each gesture
#         for gdx, name in enumerate(filenames):
#             mat = sio.loadmat(os.path.join(self.path, name))
#             emg, labels = mat['data'], mat['gesture'].ravel()
#             emg = bandpass(emg)
#             emg = bandstop(emg) ## TESTING WHETHER BANDSTOPPING IMPROVES PERFORMANCE
#             indices = get_indices(labels)
#             cur_label = int(name[4:7].lstrip('0')) -1 # get the label for the given gesture

#             # For each repetition
#             for idx in range(0, len(indices), 2):
#                 start, end = indices[idx], indices[idx+1]
#                 center = (start + end) // 2 # get the central index of the given repetition
#                 images = list(emg[center - self.fs//2 : center + self.fs//2, :].reshape(self.n_samples, 1, 8, 16, order='F'))
                
#                 # If intrasession, use given repetition for testing
#                 if intrasession and (idx+1)==test_rep:
#                     self.X_test.extend(images)
#                     self.Y_test.extend( [cur_label]*self.n_samples )
#                 else:
#                     self.X.extend(images) # add EMG surface images onto our data matrix
#                     self.Y.extend( [cur_label]*self.n_samples ) # add labels onto our label matrix
        
        
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
#             self.X = (self.X - self.min)/(self.max - self.min)*2 -1
#             if intrasession: self.X_test = (self.X_test - self.min)/(self.max - self.min)*2 -1

#         if transform:
#             self.X = transform(self.X)

#         self.len = self.n_samples * self.num_repetitions * self.num_gestures

#     def __len__(self):
#         return self.len
    
#     def train_test_split(self):
#         '''Gets a deepcopy of the current loader with test data set as training data.'''
#         test_loader = deepcopy(self) # make a copy of current loader
#         test_loader.X, test_loader.Y = deepcopy(test_loader.X_test), deepcopy(test_loader.Y_test)
#         del test_loader.X_test, test_loader.Y_test, self.X_test, self.Y_test
#         test_loader.num_repetitions = 1 # only a single repetition in the given test set
#         test_loader.len = len(self) // self.num_repetitions
#         return test_loader # returns test loader

#     def __getitem__(self, idx):
#         X, Y = self.X[idx], self.Y[idx]
#         return X, Y
    

class CSLFrameLoader(Dataset):
    def __init__(self, path='../datasets/csl', sub='subject1', transform=None, target_transform=None,
                  norm=0, num_gestures=8, num_repetitions=10, intrasession=False, session=False, test_rep=None):
        super(CSLFrameLoader, self).__init__()

        self.num_gestures = num_gestures
        self.num_repetitions = num_repetitions - int(intrasession)
        self.fs = 2048
        self.n_samples = self.fs
        self.channels = 168

        self.path = os.path.join(path, sub)
        self.transform = transform
        self.target_transform = target_transform
        self.intrasession = intrasession
        self.test_rep = test_rep

        # Initialize variables
        self.X_test, self.Y_test = [], []
        self.X, self.Y = [], []

        # If intrasession, store information for only a given session
        if intrasession:
            DIR = os.path.join(path, sub, session)
            X, Y = self.extract_frames(DIR)
            self.X.extend(X)
            self.Y.extend(Y)

        else:
            sessions = ['session{}'.format(idx) for idx in range(1, 6)]
            for idx, session in enumerate(sessions):
                DIR = os.path.join(path, sub, session)
                X, Y = self.extract_frames(DIR)
                if test_rep == (idx+1):
                    self.X_test.extend(X)
                    self.Y_test.extend(Y)
                else:
                    self.X.extend(X)
                    self.Y.extend(Y)

        # Convert data to tensor
        self.X = torch.tensor(np.array(self.X)).to(torch.float32)
        self.Y = torch.tensor(np.array(self.Y)).to(torch.float32)
        if intrasession:
            self.X_test = torch.tensor(np.array(self.X_test)).to(torch.float32)
            self.Y_test = torch.tensor(np.array(self.Y_test)).to(torch.float32)

        if norm  == 1: # standardization
            self.mean = self.X.mean()
            self.std = self.X.std()
            self.X = (self.X - self.mean)/(self.std + 1.e-12)
            if intrasession: self.X_test = (self.X_test - self.mean)/(self.std + 1.e-12)

        elif norm == -1: # scale between [-1, 1]
            self.max = self.X.amax(keepdim=True)
            self.min = self.X.amin(keepdim=True)
            self.X = (self.X - self.min)/(self.max - self.min)*2 - 1
            if intrasession: self.X_test = (self.X_test - self.min)/(self.max - self.min)*2 - 1

        if transform:
            self.X = transform(self.X)

        self.len = self.n_samples * self.num_repetitions * self.num_gestures

    def __len__(self):
        return self.len
    
    def extract_frames(self, DIR):
        ''' Extract frames for the given subject/session.'''
        filenames = os.listdir(DIR)
        X, Y = [], []
        for gdx, name in enumerate(filenames):
            mat = sio.loadmat(os.path.join(DIR, name))
            reps = mat['gestures'].shape[0] # number of repetitions stored
            cur_label = int(name.replace('gest', '').replace('.mat', '')) # get the label for the given gesture
            if cur_label == 0: continue # exclude rest
            else: cur_label -= 1 

            # For each repetition
            for idx in range(reps):
                emg = mat['gestures'][idx, 0].T
                emg = bandpass(emg, self.fs)
                emg = bandstop(emg, self.fs) 
                center = len(emg) // 2 # get the central index of the given repetition
                images = emg[center - self.fs//2 : center + self.fs//2, :]
                images = np.flip(images.reshape(self.n_samples, 1, 8, 24, order='F'), axis=0)
                images = list(images[:, :, 1:, :]) # drop first row given bipolar nature of data and create list
                
                # If intrasession, use given repetition for testing
                if self.intrasession and (idx+1)==self.test_rep:
                    self.X_test.extend(images)
                    self.Y_test.extend( [cur_label]*self.n_samples )
                else:
                    X.extend(images) # add EMG surface images onto our data matrix
                    Y.extend( [cur_label]*self.n_samples ) # add labels onto our label matrix
        return X, Y
    
    def train_test_split(self):
        '''Gets a deepcopy of the current loader with test data set as training data.'''
        test_loader = deepcopy(self) # make a copy of current loader
        test_loader.X, test_loader.Y = deepcopy(test_loader.X_test), deepcopy(test_loader.Y_test)
        del test_loader.X_test, test_loader.Y_test, self.X_test, self.Y_test
        test_loader.num_repetitions = 1 # only a single repetition in the given test set
        if self.intrasession:
            test_loader.len = len(self) // self.num_repetitions
        else:
            test_loader.len = self.n_samples * self.num_repetitions * self.num_gestures
        return test_loader # returns test loader

    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        return X, Y
