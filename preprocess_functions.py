import torch
from scipy.ndimage import median_filter
import numpy as np


# TODO
# Note that this way of preprocessing is really bad and needs to be changed ASAP
# Note again that
def compute_stats(data_matrix):
    sum1 = torch.zeros((7, 24))
    sum2 = torch.zeros((7, 24))
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            sum1 = sum1 + torch.sum(data_matrix[i][j], dim=0)
            sum2 = sum2 + torch.sum(data_matrix[i][j] ** 2, dim=0)
    return sum1, sum2

def apply_transform(data_matrix, transformation):
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            data_matrix[i][j] = transformation(data_matrix[i][j])

def apply_normalization(data_matrix, mean, std):
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            data_matrix[i][j] = (data_matrix[i][j] - mean) / (std + 1e-12)


def normalise_csl_preproc(data_matrix, total):
    sum1, sum2 = compute_stats(data_matrix)
    mean = sum1 / total
    mean_sq = sum2 / total
    std = torch.sqrt(mean_sq - mean * mean)

    apply_normalization(data_matrix, mean, std)
    return mean, std


def normalise_csl_preproc_multisub(data_list):
    total = 0
    sum1 = torch.zeros((7, 24))
    sum2 = torch.zeros((7, 24))
    for sub in data_list:
        s1, s2 = compute_stats(sub.data_matrix)
        sum1 += s1
        sum2 += s2
        total += sub.total
    mean = sum1 / total
    mean_sq = sum2 / total
    std = torch.sqrt(mean_sq - mean * mean)

    for sub in data_list:
        apply_normalization(sub.data_matrix, mean, std)
    return mean, std


def normalise_emg(signal, mean=0.0, std=1.0, eps=1e-12):
    sign_shape = signal.shape[1:]
    signal = signal.reshape(signal.shape[0], -1)  # to C, T
    mean = mean.reshape(-1)
    std = std.reshape(-1)
    signal = ((signal - mean) / (std + eps))
    signal = signal.to(torch.float32)
    if len(sign_shape) > 1:
        return signal.reshape(signal.shape[0], sign_shape[0], sign_shape[1])
    return signal

# TODO this has to be merged and a consistent way to deal with all these has to be decided on.
def normalise_emg_correct(signal, mean=0.0, std=1.0, eps=1e-12):
    signal = (signal - mean)/ (std + 1.e-12)
    return signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y


def median(data, num_semg_row, num_semg_col):
    return np.array([median_filter(image, 3).ravel() for image
                     in data.reshape(-1, num_semg_row, num_semg_col)])


def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
    for begin, end in zip([0] + breaks, breaks + [len(label)]):
        assert begin < end
        yield begin, end


def _csl_cut(data, framerate):
    from scipy.ndimage import median_filter
    window = int(np.round(150 * framerate / 2048))
    data = data[:len(data) // window * window].reshape(-1, 150, data.shape[1])
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)]
    rms = np.mean(rms, axis=1)
    threshold = np.mean(rms)
    mask = rms > threshold
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    begin, end = max(continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0]))
    return begin * window, end * window