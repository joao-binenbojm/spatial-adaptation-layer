from collections import Counter
from scipy import signal
import numpy as np


def get_rms_signal(emg, M=32, s=1):
    '''Computes the instantaneous estimate of RMS from a window of L samples. Returns a signal of the same length.'''
    emg_square = np.square(emg)
    ma_filter = np.ones((2*M + 1, 1)) / (2*M+1) # computes the average within the given window
    ms_signal = signal.convolve(emg_square, ma_filter, mode='same')
    rms_signal = np.sqrt(ms_signal)
    return rms_signal

## Majority Voting
def majority_voting(predictions, M=32):
    ''' Go over the length of the segment, and sequentially increase the count of a given symbol.
        This count is then removed when the window moves out of its location. This results in only
        tracking the counts of elements within the given window. Majority voting window is centered around
        present index, with M past and M future indices.
        Inputs:
            arr[ndarray]: input segment to be majority voted over
            M[int]: majority voting window size
        Returns:
            modes[list]: contains labels after majority voting windows
    '''
    counts = Counter()  # Count occurrences in the first W-1 elements
    modes = []  # List to store the modes for each window
    L = len(predictions)
    for i in range(-M, L): # goes up by 1 every time
        # Only begin reducing counts one index after a full voting window
        if i > M:
            counts[predictions[i - M - 1]] -= 1  # Decrease count of the element leaving the window
            if counts[predictions[i - M]] == 0:
                del counts[predictions[i - M]]  # Remove element from counts if count becomes 0
        # Increase count of (i+M)th element while there is still data left
        if i + M < L:
            counts[predictions[i + M]] += 1
        # Only begin adding modes from 0th index
        if i >= 0:
            mode = max(counts, key=counts.get)  # Calculate the mode for the current window
            modes.append(mode)

    return modes

def majority_voting_segments(predictions, M=32, n_samples=1000):
    ''' This function assumes that multiple repetition segments are concatenated together.
        Thus, it applies majority voting to each segment individually.
    '''
    voted_predictions = []
    for idx in range(len(predictions) // n_samples):
        preds = predictions[idx*n_samples : (idx+1)*n_samples] # extract segment
        voted_predictions_segment = majority_voting(preds, M=M) # get majority voting of given segment
        voted_predictions.extend(voted_predictions_segment) # add majority voted to final segment
    return voted_predictions

## FILTERING
identity = lambda x, fs: x

def bandstop(data, fs=1000):
    '''Used to remove powerline interference and its multiples.'''
    sos = signal.butter(2, (45, 55), btype='bandstop', output='sos', fs=fs)
    data = signal.sosfilt(sos, data, axis=0)
    return data

def bandpass(data, fs=1000):
    '''Used to maintain only information in relevant anatomical range of sEMG activity.'''
    sos = signal.butter(2, (20, 380), btype='bandpass', output='sos', fs=fs)
    data = signal.sosfilt(sos, data, axis=0)
    return data