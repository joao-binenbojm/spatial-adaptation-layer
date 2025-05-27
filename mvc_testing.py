import wfdb
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from emg_processing import get_rms_signal


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


DIR = '/home/joao/Desktop/datasets/hyser/mvc_dataset'
subdir = 'subject01_session1'

# Load files
files = [file.replace('.hea', '').replace('.dat', '') for file in os.listdir(os.path.join(DIR, subdir))]
files = list(set(files)) # only keep record names

# Save force profiles for each file
files = [file for file in files if 'force' in file]
baseline = np.zeros(256)
baseline_samp_count = 0
for file in files:
    record = wfdb.rdrecord(os.path.join(DIR, subdir, file))
    num_finger = int(file.split('finger')[1][0])
    force = np.abs(record.p_signal.mean(axis=1))
    force = upsample_signal(force, up=2048, down=100)
    force_off = (force < 0.2*force.max()).astype(int)
    force_off_processed = process_binary_signal(force_off)
    force = (force - np.min(force)) / (np.max(force) - np.min(force))

    # Get RMS of EMG and compare to force profile
    emg_record = wfdb.rdrecord(os.path.join(DIR, subdir, file.replace('force', 'raw')))
    emg = emg_record.p_signal
    emg = emg - emg.mean(axis=0) # make zero mean signals
    emg = emg[force_off_processed == 0] # remove force off segments
    baseline += (emg**2).sum(axis=0)
    baseline_samp_count += emg.shape[0]

    plt.plot(force)
    plt.plot(force_off)
    plt.plot(force_off_processed)
    plt.title(file)
    plt.savefig(os.path.join('sal_classification/tmp', file))
    plt.close()

# Plot baseline
baseline /= baseline_samp_count
baseline = baseline.reshape(1,-1)

ngrids = 4
subimages = []
for grid_idx in range(ngrids):
    subimage = np.array(baseline[:, grid_idx*64:(grid_idx+1)*64]).reshape(baseline.shape[0], 1, 8, 8)
    subimage = np.flip(np.flip(subimage, axis=2), axis=3)
    subimages.append(subimage)
images = np.concatenate(subimages, axis=2) # append along horizontal direction

plt.figure()
plt.imshow(images[0, 0], cmap='gray')
plt.title('Baseline')
plt.savefig(os.path.join('sal_classification/tmp', 'baseline'))
plt.close()
