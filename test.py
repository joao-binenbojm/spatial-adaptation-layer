import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
from sal_decomposition.utils import utils


with open("/home/joao/Desktop/spatial_adaptation_layer/output_data.pkl", 'rb') as file:
    loaded_dict = pickle.load(file)

emg_grid_test = loaded_dict['emg_grid_test']
pred_dts = loaded_dict['pred_dts']
mu_dts2 = loaded_dict['mu_dts2']

n1, n2 = 5-1, 12-1
mu1 = np.zeros(emg_grid_test.shape[0])
mu1[pred_dts[n1].astype(int)] = 1.0

mu2 = np.zeros(emg_grid_test.shape[0])
mu2[mu_dts2[n2].astype(int)] = 1.0


# Parameters for computing f1 score
fs = 2048
jitter = 0.002


matches, f1_scores, sensitivities, precisions = utils.spike_matching(mu_dts2, pred_dts, fs=fs, jitter=jitter)

# # Compute matches using broadcasting
# time_diffs = torch.abs(torch.tensor(pred_dts[n1].reshape(-1, 1)) - torch.tensor(mu_dts2[n2].reshape(1, -1)))  # Shape: (n_pred, n_true)
# matches = (time_diffs <= int(jitter*fs)).any(dim=1)  # Assuming spike_match_jitter threshold is 1
# tps = matches.sum()
# fps = len(pred_dts[n1]) - tps

# # Compute false negatives using vectorized operations
# gt_matches = (time_diffs <= int(jitter*fs)).any(dim=0)
# fns = (~gt_matches).sum()

# sensitivity = tps / (tps + fns)
# precision = tps / (tps + fps)

# f1_scores = 2 * sensitivity * precision / (sensitivity + precision + 1e-12)

# print()

## PLOTTING ONE UNIT VS REST FROM OTHER SESSION 
# for idx in range(len(mu_dts2)):
#     mu2 = np.zeros(emg_grid_test.shape[0])
#     mu2[mu_dts2[idx].astype(int)] = 1.0

#     plt.figure()
#     plt.plot(mu1[15000:18000], 'b')
#     plt.plot(mu2[15000:18000], 'r--')
#     plt.title(f'MU #{n1+1} VS. MU #{idx+1}')
#     plt.savefig('spikes')
#     sleep(3)
#     plt.close()