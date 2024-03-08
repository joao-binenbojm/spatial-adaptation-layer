import os

import scipy.io as sio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import preprocess_functions as preprocess_functions


# File structure
# 7 classes
# CSL , CGM
# Windowed vs Non-windowed
# Single subject vs Multi Subject


class CapGMyoLoader(Dataset):
    def __init__(self, path='./CapGMyo', db='dbb',train=True, transform=None, target_transform=None,
                 subject_list=[1], window=64, stride=1, norm=0, num_gestures=8, train_repetitions=10, intrasession=False  ):
        super(CapGMyoLoader, self).__init__()

        self.num_gestures = num_gestures
        self.num_repetitions = 10
        self.train_repetitions = train_repetitions
        self.samples = 1000
        self.channels = 128

        self.num_subjects = len(subject_list)

        self.path = os.path.join(path, db)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train
        subject_list.sort()
        self.subject_list = subject_list

        self.window = window
        self.stride = stride



        data_emg = []
        labels = []
        for subject in self.subject_list:
            for i in range(self.num_gestures):
                for j in range(self.num_repetitions):
                    file_ext = f'{subject:03}-{i + 1:03}-{j + 1:03}.mat'
                    mat = sio.loadmat(os.path.join(self.path, file_ext))
                    data_emg.append(mat['data'])
                    labels.append(mat['gesture'][0][0])

        self.data_tensor = np.stack(data_emg)
        self.label_tensor = np.stack(labels)

        self.data_tensor = torch.tensor(self.data_tensor).to(torch.float32).reshape(self.num_subjects, self.num_gestures,
                                                                        self.num_repetitions, self.samples, self.channels)
        self.label_tensor = torch.tensor(self.label_tensor)

        # if intrasession is specified, remove some of the gesture
        if intrasession:
            if train:
                self.data_tensor = self.data_tensor[:, :, :self.train_repetitions, :, :]
                self.num_repetitions = self.train_repetitions
            else:
                self.data_tensor = self.data_tensor[:, :, self.train_repetitions:,  :, :]
                self.num_repetitions = self.num_repetitions - self.train_repetitions



        if norm  == 1:
            self.mean = self.data_tensor.mean(dim=[1,2,3], keepdim=True)
            self.std = self.data_tensor.std(dim=[1,2,3], keepdim=True)

            self.data_tensor = (self.data_tensor - self.mean)/(self.std + 1.e-12)
        elif norm == -1:
            self.max = self.data_tensor.amax(dim=[0,1,2,3], keepdim=True)
            self.min = self.data_tensor.amin(dim=[0,1,2,3], keepdim=True)

            self.data_tensor = (self.data_tensor - self.min)/(self.max - self.min)*2 -1

        if transform:
            self.data_tensor = transform(self.data_tensor)

        self.len = self.num_subjects * self.num_gestures * self.num_repetitions * self.samples // self.stride

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        samples_per_repetition = self.samples // self.stride

        sample = idx % samples_per_repetition
        idx //= samples_per_repetition

        repetition = idx % self.num_repetitions
        idx //= self.num_repetitions

        gesture = idx % self.num_gestures
        idx //= self.num_gestures

        subject = idx % self.num_subjects

        # find window according to index
        window_middle = sample * self.stride
        right_lim = window_middle + self.window // 2
        left_lim = window_middle - self.window // 2
        point = self.data_tensor[subject, gesture, repetition, max(left_lim, 0):min(right_lim, self.samples), :]


        # pad the data on the left and right
        left_padding_len = -left_lim if left_lim < 0 else 0
        right_padding_len = right_lim - self.samples if right_lim > self.samples else 0
        point = F.pad(point, (0, 0, left_padding_len, right_padding_len), "constant", 0)

        return point.T, gesture


import os

import scipy.io as sio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import preprocess_functions as preprocess_functions

# currently this only computes stats for each separate subject
#TODO change this
class CSLSMultiSubjectWindow(Dataset):
    def __init__(self, path='./CSL', train=True, transform=None, target_transform=None, window=160, stride=8, \
                 subject_list=[1], sessions=[1], num_gestures=27, downsample=1, norm=False):

        super(CSLSMultiSubjectWindow, self).__init__()
        self.num_gestures = num_gestures

        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        self.sessions = sessions
        self.m_sessions = len(sessions)
        self.train = train

        self.data_list = []
        self.n_subs = len(subject_list)

        total = 0
        self.slice_list = [0]
        for i in range(len(subject_list)):
            self.data_list.append(CSLSubjectPreprocWindow(path, subject=subject_list[i],
                                                          train=train, sessions=sessions,
                                                          num_gestures=num_gestures,
                                                          transform=transform, downsample=downsample,
                                                          window=window, stride=stride, norm=norm))
            total += len(self.data_list[i])
            self.slice_list.append(total)

        self.data_size = total
        # if norm:
        #     stats = torch.stack([preprocess_functions.normalise_csl_preproc(subject_item.data_matrix, subject_item.total) for subject_item in self.data_list])

        if transform:
            for sub in self.data_list:
                preprocess_functions.apply_transform(sub.data_matrix, transform)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        for i in range(1, len(self.slice_list)):
            if self.slice_list[i] > idx:
                idx -= self.slice_list[i - 1]
                break
        return self.data_list[i - 1][idx]


class CSLSubjectPreprocWindow(Dataset):
    def __init__(self, path='./CSL/', train=True, transform=None, target_transform=None, \
                 subject=1, sessions=[1], num_gestures=27, downsample=1, window=64, stride=1, return_2d=False,
                 norm=False):
        super(CSLSubjectPreprocWindow, self).__init__()

        self.num_gestures = num_gestures

        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        self.sessions = sessions
        self.m_sessions = len(sessions)
        self.train = train

        self.slice_map = [0]
        prev_slice = 0

        sessions.sort()

        self.window = window
        self.stride = stride
        self.return_2d = return_2d

        self.data_matrix = [[0 for s in sessions] for i in range(num_gestures)]

        for i in range(num_gestures):
            for s in range(len(sessions)):
                # iterate for 10 repetitions
                file_ext = f'subject{subject:01}/session{sessions[s]:01}/gest_preproc{i:01}.pt'
                self.data_matrix[i][s] = torch.load(os.path.join(path, file_ext)).reshape(-1, 7, 24)
                self.data_matrix[i][s] = torch.tensor(np.float32(self.data_matrix[i][s]))
                if downsample > 1:
                    self.data_matrix[i][s] = self.data_matrix[i][s][::downsample]

                l = self.data_matrix[i][s].shape[0] // self.stride

                prev_slice += l
                self.slice_map.append(prev_slice)

        self.total = prev_slice

        if norm:
            self.mean, self.std = preprocess_functions.normalise_csl_preproc(self.data_matrix, self.total)

        if transform:
            preprocess_functions.apply_transform(self.data_matrix, transform)

        # Put time as the last dimension
        for i in range(num_gestures):
            for s in range(len(sessions)):
                self.data_matrix[i][s] = self.data_matrix[i][s].transpose(0, 1)
                self.data_matrix[i][s] = self.data_matrix[i][s].transpose(1, 2)
        if not return_2d:
            preprocess_functions.apply_transform(self.data_matrix, lambda t: t.reshape(7 * 24, -1))

    def __len__(self):
        return self.total

    def binary_lower_bound(arr, low, high, x):
        # Check base case
        if arr[high] <= x:
            return high
        if high - low <= 1:
            return low
        if high >= low:

            mid = (high + low) // 2

            if arr[mid] == x:
                return mid
            elif arr[mid] > x:
                return CSLSubjectPreprocWindow.binary_lower_bound(arr, low, mid - 1, x)

            else:
                return CSLSubjectPreprocWindow.binary_lower_bound(arr, mid, high, x)

        else:
            return low

    def __getitem__(self, idx):

        index = CSLSubjectPreprocWindow.binary_lower_bound(self.slice_map, 0, len(self.slice_map) - 1, idx)
        offset = idx - self.slice_map[index]

        gesture_id = index // self.m_sessions
        session_id = index % self.m_sessions

        segment_size = self.data_matrix[gesture_id][session_id].shape[-1]

        window_middle = offset * self.stride
        right_lim = window_middle + self.window // 2
        left_lim = window_middle - self.window // 2
        if self.return_2d:
            point = self.data_matrix[gesture_id][session_id][:, :, max(left_lim, 0):  min(right_lim, segment_size)]
        else:
            point = self.data_matrix[gesture_id][session_id][:, max(left_lim, 0):  min(right_lim, segment_size)]
            # pad the data on the left and right
        left_padding_len = -left_lim if left_lim < 0 else 0
        right_padding_len = right_lim - segment_size if right_lim > segment_size else 0
        point = F.pad(point, (left_padding_len, right_padding_len), "constant", 0)

        return point, gesture_id


# TODO
# Consider transforming data into tensors instead of keeping them as numpy array
class CSLSubjectPreproc(Dataset):
    def __init__(self, path='./CSL/', train=True, transform=None, target_transform=None, \
                 subject=1, sessions=[1], num_gestures=27, downsample=1, norm=False):
        super(CSLSubjectPreproc, self).__init__()

        self.num_gestures = num_gestures

        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        self.sessions = sessions
        self.m_sessions = len(sessions)
        self.train = train

        self.slice_map = [0]
        prev_slice = 0

        sessions.sort()

        self.data_matrix = [[0 for s in sessions] for i in range(num_gestures)]

        for i in range(num_gestures):
            for s in range(len(sessions)):
                # iterate for 10 repetitions
                file_ext = f'subject{subject:01}/session{sessions[s]:01}/gest_preproc{i:01}.pt'
                self.data_matrix[i][s] = torch.load(os.path.join(path, file_ext)).reshape(-1, 7, 24)
                self.data_matrix[i][s] = torch.tensor(np.float32(self.data_matrix[i][s]))

                if downsample > 1:
                    self.data_matrix[i][s] = self.data_matrix[i][s][::downsample]
                l = self.data_matrix[i][s].shape[0]
                prev_slice += l
                self.slice_map.append(prev_slice)

        self.total = prev_slice
        if norm:
            self.mean, self.std = preprocess_functions.normalise_csl_preproc(self.data_matrix, self.total)
        if transform:
            preprocess_functions.apply_transform(self.data_matrix, transform)

    def __len__(self):
        return self.total

    def binary_lower_bound(arr, low, high, x):
        # Check base case
        if arr[high] <= x:
            return high
        if high - low <= 1:
            return low
        if high >= low:

            mid = (high + low) // 2

            if arr[mid] == x:
                return mid
            elif arr[mid] > x:
                return CSLSubjectPreproc.binary_lower_bound(arr, low, mid - 1, x)

            else:
                return CSLSubjectPreproc.binary_lower_bound(arr, mid, high, x)

        else:
            return low

    def __getitem__(self, idx):

        index = CSLSubjectPreproc.binary_lower_bound(self.slice_map, 0, len(self.slice_map) - 1, idx)
        offset = idx - self.slice_map[index]

        gesture_id = index // self.m_sessions
        session_id = index % self.m_sessions
        return self.data_matrix[gesture_id][session_id][offset], gesture_id


class CSLSMultiSubject(Dataset):
    def __init__(self, path='./CSL', train=True, transform=None, target_transform=None, \
                 subject_list=[1], sessions=[1], num_gestures=27, downsample=1, norm=False):
        self.num_gestures = num_gestures

        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        self.sessions = sessions
        self.m_sessions = len(sessions)
        self.train = train

        self.data_list = []
        self.n_subs = len(subject_list)

        total = 0
        self.slice_list = [0]
        for i in range(len(subject_list)):
            self.data_list.append(CSLSubjectPreproc(path, subject=subject_list[i],
                                                    train=train, sessions=sessions,
                                                    num_gestures=num_gestures,
                                                    transform=transform, downsample=downsample, norm=False))
            total += len(self.data_list[i])
            self.slice_list.append(total)

        self.data_size = total
        if norm:
            self.mean, self.std = preprocess_functions.normalise_csl_preproc_multisub(self.data_list)

        if transform:
            for sub in self.data_list:
                preprocess_functions.apply_transform(sub.data_matrix, transform)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        for i in range(1, len(self.slice_list)):
            if self.slice_list[i] > idx:
                idx -= self.slice_list[i - 1]
                break
        return self.data_list[i - 1][idx]





# Custom sampler for Adaptive batch normalization according to CapGMyo Paper
from torch.utils.data import Sampler


class CustomShuffleSampler(Sampler):
    def __init__(self, data_source, custom_shuffle_function):
        self.data_source = data_source
        self.custom_shuffle_function = custom_shuffle_function

    def __iter__(self):
        self.indices = list(range(len(self.data_source)))
        self.shuffled_indices, self.subs_id = self.custom_shuffle_function(self.indices)
        return iter(self.shuffled_indices)

    def __len__(self):
        return len(self.data_source)


def compute_batch_subject(indices):
    return [indices[i] // 80 for i in range(len(indices))]


def custom_shuffle(indices):
    import random
    # split per subject
    sliced_array = [indices[i * 80000:(i + 1) * 80000] for i in range(len(indices) // 80000)]
    # shuffle
    for i in range(len(sliced_array)):
        random.shuffle(sliced_array[i])

    # reconnect array
    tmp = []
    for i in range(len(sliced_array)):
        tmp = tmp + sliced_array[i]

    # split per batch and shuffle
    sliced_array = [tmp[i * 1000:(i + 1) * 1000] for i in range(len(tmp) // 1000)]
    batch_indices = list(range(len(sliced_array)))
    random.shuffle(batch_indices)

    final = []
    for i in range(len(sliced_array)):
        final = final + sliced_array[batch_indices[i]]

    subjects_id = compute_batch_subject(batch_indices)
    return final, subjects_id

# Example usage:
# Assuming you have a custom dataset named 'my_dataset' and it's initialized properly






# Delsys loader

class DTLoader(Dataset):
    def __init__(self, path='./pilot.csv', db='dbb', train=True, transform=None,
                 target_transform=None, window=256, stride=8, norm=True, intrasession=False):
        super(DTLoader, self)
        self.path = os.path.join(path, db)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train

        self.window = window
        self.stride = stride

        total_reps = 40
        if intrasession:
            total_reps = 80

        # compute labels positions
        data_raw = np.genfromtxt(path, delimiter=',')[1:, 1:]
        labels_raw = data_raw[:,8]
        labels_rolled = np.roll(np.copy(labels_raw), 1)
        delta = (labels_raw - labels_rolled)
        indices = np.where(delta != 0)[0][:]

        self.interval_size = 1000

        middle_segments = []
        labels_seg = []
        for i in range(len(indices) // 2):
            ci = 2*i
            start = indices[ci]
            end = indices[ci + 1]
            middle = (start + end) // 2

            seg = data_raw[middle - self.interval_size//2:middle + self.interval_size//2,0:8]

            middle_segments.append(seg)
            labels_seg.append(data_raw[middle][8])


        self.data  = np.stack(middle_segments)
        self.labels = np.stack(labels_seg)

        self.data = self.data.reshape(8, 10, self.interval_size, 8)
        self.labels = self.labels.reshape(8, 10)
        #print(self.data.shape)

        if not intrasession:
            if train:
                self.data = self.data[:, :5,:,:]
                self.labels = self.labels[:, :5]
            else:
                self.data = self.data[:, 5:,:, :]
                self.labels = self.labels[:, 5:]


        self.data = self.data.reshape(total_reps * self.interval_size, 8)
        self.data = self.data.T
        self.labels = torch.tensor(self.labels.reshape(-1)).to(torch.long) - 1

        self.data = torch.tensor(self.data).to(torch.float32)
        self.len = total_reps * (self.interval_size // self.stride)

        if norm:
            self.mean = self.data.mean(dim=1)[:, None]
            self.std = self.data.std(dim=1)[:, None]
            self.data = (self.data - self.mean) / (self.std + 1.e-12)

        if transform:
            self.data = transform(self.data)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        samples_per_session = self.interval_size // self.stride
        session_start = (idx // samples_per_session) * self.interval_size

        window_middle = (idx % samples_per_session) * self.stride
        right_lim = window_middle + self.window // 2
        left_lim = window_middle - self.window // 2
        point = self.data[:, session_start + max(left_lim, 0):session_start + min(right_lim, self.interval_size)]
        left_padding_len = -left_lim if left_lim < 0 else 0
        right_padding_len = right_lim - self.interval_size if right_lim > self.interval_size else 0
        point = F.pad(point, (left_padding_len, right_padding_len), "constant", 0)

        return point, self.labels[idx // samples_per_session]