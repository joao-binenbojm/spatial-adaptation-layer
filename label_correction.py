import numpy as np
import scipy
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import LinAlgError

## This script is used to implement the GML algorithm to better match labels to movement
def normal_pdf(X, mus, sigmas, tol=1e-10):
    '''Assumes every column of X is a new observation.'''
    D, N = X.shape # number of observations in X matrix
    mus = mus.reshape(-1, 1) # reshape as column vector
    sigmas_det = np.linalg.det(sigmas) # covariance matrix determinant
    if np.abs(sigmas_det) > tol: # threshold for checking if matrix is singular
        sigmas_inv = np.linalg.inv(sigmas) # inverse covariance matrix
        exponent = np.array([-0.5*(X[:,[idx]] - mus).T @ sigmas_inv @ (X[:,[idx]] - mus) for idx in range(N)])
        densities = (np.exp(exponent) / np.sqrt( (2*np.pi)**D * sigmas_det)).flatten()
        return densities # N densities
    else:
        raise LinAlgError

def label_correction(labels, movement, fs=2000):
    ''' 
    Takes in labels, movement, and provides the corrected labels,
    as in Ilja Kuzborskij et al., 2012
    '''
    diffs = np.diff(labels, append=np.zeros(1))
    tstarts = np.argwhere(diffs > 0).flatten().astype(int) # when movements begin
    tstops = np.argwhere(diffs < 0).flatten().astype(int) # when movements stop
    stride = int(fs*0.01) # 10ms, as in original paper
    tstartstop = [] # contains tuples with start/stop times of movements 
    
    # Begin iterating over each movement performed
    for tstart, tstop in tqdm(zip(tstarts, tstops)):
        tstop_delay = tstop + int(fs*1) # allows for 1s of movement delay
        t_pairs = [] # keep track of t0 and t1 in tuples
        likelihoods = [] # keep track of max likelihood for each pair
        for t0 in tqdm(range(tstart+int(fs*0.1), int(tstop*0.7 + fs*1), stride)): # 0.1s minimum delay considered
            for t1 in range(t0 + int(0.3*(tstop-tstart)), tstop_delay, stride): # adding two constraints from paper
                # MLE solutions for mean and covariances
                sgn0 = np.vstack((movement[tstart:t0,:], movement[t1:tstop_delay])) # rest signal
                sgn1 = movement[t0:t1] # movement signal
                means0, means1 = sgn0.mean(axis=0), sgn1.mean(axis=0) # MLE mean estimate
                cov0, cov1 = np.cov(sgn0, rowvar=False), np.cov(sgn1, rowvar=False) # compute MLE covariance estimates
                # if np.trace(cov1) > np.trace(cov0): # if signal 1 has greater variance
                    # # Compute overall likelihoods!
                    # if np.linalg.det(cov0) < 1e-10:
                    #     print('WTF')
                densities0 = normal_pdf(sgn0.T, means0, cov0)
                densities1 = normal_pdf(sgn1.T, means1, cov1)
                L = np.log(densities0).sum() + np.log(densities1) # compute likelihood
                likelihoods.append(L)
                t_pairs.append((t0, t1))
        # Store t_pair that led to largest likelihood value
        max_idx = np.argmax(likelihoods)
        tstartstop.append(t_pairs[max_idx]) # store timing variables with highest likelihoods
    # Compute new label signal based on t_pairs
    new_labels = []
    for t0, t1 in tstartstop:
        label = scipy.stats.mode(labels[t0:t1])[0] # what label the current period corresponds to
        new_labels.extend([0]*(t0 - len(new_labels))) # add zeros for rest
        new_labels.extend([label]*(t1 - t0)) # add labels for movement
    new_labels.extend([0]*(len(labels) - len(new_labels))) # add final rest samples
    ##########
    ## ADD METHOD TO QUANTIFY THE SIZE OF THE FIX as a metric
    ##########

    return np.array(new_labels)


if __name__ == '__main__':
    # Simulate data
    T = 100 # 100s recording
    fs = 2000 # sampling rate
    Nch = 12 # number of channels
    tchange = [0]
    labels, movements, movement_labels = [], [], [] # movement_labels are the ground truth
    # when the subject really began the movement

    # Get change points between movement and rest
    for idx in range(int(T/8)): # roughly how many movements will fit in 100s
        tchange.append(tchange[-1] + 3*fs)
        tchange.append(tchange[-1] + 5*fs)
    tchange.append(tchange[-1] + 3*fs) # add extra buffer of rest a tthe end

    # Create movement and label signals (with some movement delays)
    for tdx in range(1, len(tchange)): # while we can still add more movements within 100s
        if tdx < len(tchange) - 1:
            delay = random.randint(int(0.25*fs), int(0.5*fs)) # approximate human reaction times
        else:
            delay = 0 # add 0 delay for last iteration

        tlabel = tchange[tdx] - len(labels) # get number of samples for label
        tmovement = tchange[tdx] - len(movements) + delay # number of samples for movement
        label = int(tdx % 2 == 0) # alternates between 0 and 1 (rest/movement)
        labels.extend([label]*(tlabel))
        movement_labels.extend([label]*(tmovement))
        # Create signal with WGN, such that the means are difference and movement has more std
        movements.extend(np.random.normal(label, (label + 1)/2, (tmovement, Nch)).tolist())
            
    labels, movements, movement_labels = np.array(labels), np.array(movements), np.array(movement_labels)

    # Plotting simulated label misalignment
    # plt.figure()
    # plt.plot(movement_labels)
    # plt.plot(labels)
    # plt.show()

    print(np.mean(movement_labels == labels)) # Measure of signal alignment

    ## Evaluate label correction algorithm:
    new_labels = label_correction(labels, movements)
    plt.figure()
    plt.plot(movement_labels)
    plt.plot(new_labels)
    plt.show()


