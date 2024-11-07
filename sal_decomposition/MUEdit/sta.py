import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
fileName = "decomposition_output.pkl"
fsamp = 10000
MUAP = []
 
 
def gausswin(M, alpha=2.5):
    """Python equivalent of the in-built gausswin function MATLAB (since there is no open-source Python equivalent)"""
 
    n = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1, dtype=np.float64)
    w = np.exp((-1 / 2) * (alpha * n / ((M - 1) / 2)) ** 2)
    return w
 
 
def cutMUAP(MUPulses, length, Y):
    """Direct conversion of MATLAB code in-lab. Extracts consecutive MUAPs out of signal Y and stores
    them row-wise in the out put variable MUAPs.
    Inputs:
    - MUPulses: Trigger positions (in samples) for rectangualr window used in extraction of MUAPs
    - length: radius of rectangular window (window length = 2*len +1)
    - Y: Single signal channel (raw vector containing a single channel of a recorded signals)
    Outputs:
    - MUAPs: row-wise matrix of extracted MUAPs (algined signal intervals of length 2*len+1)
    """
 
    while len(MUPulses) > 0 and MUPulses[-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]
 
    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = gausswin(
        2 * edge_len
    )  # gives the same output as the in-built gausswin function in MATLAB
    # create the filtering window
    win = np.ones(2 * length + 1)
    win[:edge_len] = tmp[:edge_len]
    win[-edge_len:] = tmp[edge_len:]
    MUAPs = np.empty((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length, 1) - (MUPulses[k] - length)
        end = MUPulses[k] + length - min(MUPulses[k] + length, len(Y))
        MUAPs[k, :] = win * np.concatenate(
            (
                np.zeros(start),
                Y[max(MUPulses[k] - length, 1) : min(MUPulses[k] + length, len(Y)) + 1],
                np.zeros(end),
            )
        )
 
    return MUAPs
 
 
with open(fileName, "rb") as file:
 
    output = pickle.load(file)
 
    firings = np.zeros((np.shape(output["pulse_trains"][0])))
    nmus = np.shape(output["pulse_trains"][0])[0]
 
    for i in range(nmus):
 
        firings[i, output["discharge_times"][0][i]] = 1
        MUAPacrosschans = []
 
        for j in range(64):
 
            MUAPacrosschans.append(
                cutMUAP(
                    output["discharge_times"][0][i],
                    int(fsamp * 0.02),
                    output["data"][j, :],
                )
            )
 
        MUAP.append(MUAPacrosschans)
 
    # clean the firings
    removal_condition = output["pulse_trains"][0][0, :] < 0.05
    firings = firings[:, ~removal_condition]
    distimes = []
    MUAPa = []
 
    for i in range(nmus):
 
        MUAPacrosschans = []
        distimes.append(np.where(firings[i, :] == 1)[0])
 
        for j in range(64):
 
            MUAPacrosschans.append(np.mean(MUAP[i][j], axis=0).T)
 
        MUAPa.append(MUAPacrosschans)
 
    print("MUAPs determined")
 
    # Assuming MUAPa is a NumPy array with shape (8, chan, time)
    # For example:
    # MUAPa = np.random.rand(8, 16, 100)  # Replace with actual data
 
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()
    nplot = 0
 
    for i in range(1):  # Loop over 8 cells
        ax = axes[nplot]  # Get the subplot axes
        for j in range(13):  # Loop over 16 channels
            ax.plot(MUAPa[i][j] - (j * 20))
 
        nplot += 1
 
    plt.tight_layout()
    plt.show()
 
 