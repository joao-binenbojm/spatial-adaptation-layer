import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sal_decomposition.MUEdit.processing_tools import extend_emg, whiten_emg, get_silohuette, maxk

# Arnault's matrix reshaping
index_matrix = np.array([[63, 38, 37, 12, 11, 63, 38, 37, 12, 11], # ankle
                [62, 39, 36, 13, 10, 62, 39, 36, 13, 10],
                [61, 40, 35, 14,  9, 61, 40, 35, 14,  9],
                [60, 41, 34, 15,  8, 60, 41, 34, 15,  8],
                [59, 42, 33, 16,  7, 59, 42, 33, 16,  7],
                [58, 43, 32, 17,  6, 58, 43, 32, 17,  6],
                [57, 44, 31, 18,  5, 57, 44, 31, 18,  5],
                [56, 45, 30, 19,  4, 56, 45, 30, 19,  4],
                [55, 46, 29, 20,  3, 55, 46, 29, 20,  3],
                [54, 47, 28, 21,  2, 54, 47, 28, 21,  2],
                [53, 48, 27, 22,  1, 53, 48, 27, 22,  1],
                [52, 49, 26, 23,  0, 52, 49, 26, 23,  0],
                [51, 50, 25, 24,  0, 51, 50, 25, 24,  0],
                [0, 24, 25, 50, 51,  0, 24, 25, 50, 51],
                [0, 23, 26, 49, 52,  0, 23, 26, 49, 52],
                [1, 22, 27, 48, 53,  1, 22, 27, 48, 53],
                [2, 21, 28, 47, 54,  2, 21, 28, 47, 54],
                [3, 20, 29, 46, 55,  3, 20, 29, 46, 55],
                [4, 19, 30, 45, 56,  4, 19, 30, 45, 56],
                [5, 18, 31, 44, 57,  5, 18, 31, 44, 57],
                [6, 17, 32, 43, 58,  6, 17, 32, 43, 58],
                [7, 16, 33, 42, 59,  7, 16, 33, 42, 59],
                [8, 15, 34, 41, 60,  8, 15, 34, 41, 60],
                [9, 14, 35, 40, 61,  9, 14, 35, 40, 61],
                [10, 13, 36, 39, 62, 10, 13, 36, 39, 62],
                [11, 12, 37, 38, 63, 11, 12, 37, 38, 63]]) # knee

# In the order of the cables, it is
# GRID 4    GRID 3
# GRID 1    GRID 2
# So taking the 256 signals in signal.data as input, one must reshape in
# the following way:

index_matrix[13:26,5:10] =  index_matrix[13:26,5:10] + 64 
index_matrix[0:13,5:10] = index_matrix[0:13,5:10] + 64 + 64 
index_matrix[0:13,0:5] = index_matrix[0:13,0:5] + 64 + 64 + 64   

def get_inv_cov(signal, explained_var=0.99):

    """ Get inverse of covariance of extended EMG signal with eigenvalue truncation for regularization. """
    cov_mat = np.cov(np.squeeze(signal),bias=True)
    print('FINISHED GETTING COVARIANCE MATRIX...')
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eigh(cov_mat)
    print('FINISHED GETTING EIGENDECOMPOSITION...')
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)

    # penalty = np.mean(sorted_evalues[len(sorted_evalues)//2:]) # int won't wokr for odd numbers
    # penalty = max(0, penalty)

    # rank_limit = np.sum(evalues > penalty)-1
    # if rank_limit < np.shape(signal)[0]:

    #     hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2
    # # use the rank limit to segment the eigenvalues and the eigenvectors
    # evectors = evectors[:,evalues > hard_limit]
    # evalues = evalues[evalues>hard_limit]
    # sorted_evalues = np.sort(evalues)[::-1]
    sorted_idxs = np.argsort(evalues)[::-1] # sort in descending order
    evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
    cum_explained_var = evalues.cumsum() / evalues.sum()
    evalues, evectors = evalues[cum_explained_var <= explained_var], evectors[:, cum_explained_var <= explained_var]

    inv_cov = evectors @ np.diag(1 / (evalues)) @ np.transpose(evectors)
    return inv_cov

def open_mat_output(DIR, name):
    ''' Open data from Emanuele's files so that it can be further processed'''
    arr = loadmat(os.path.join(DIR, name), struct_as_record=False, mat_dtype=True)
    
    # Load acquisition signal
    signal, edition = {}, {}
    for attr in dir(arr['signal'][0,0]):
        if attr[0] != '_': # only data relevant attributes
            signal[attr] = getattr(arr['signal'][0,0], attr)

    for attr in dir(arr['edition'][0,0]):
        if attr[0] != '_': # only data relevant attributes
            edition[attr] = getattr(arr['edition'][0,0], attr)
    
    return signal, edition

    # # Load acquisition signal
    # signal = {}
    # signal['nchans'] = arr['signal'][0,0][0].shape[0]
    # signal['fsamp'] = arr['signal'][0,0][2][0,0]
    # # signal['ngrids'] = arr['signal'][0,0][4][0,0]
    # signal['path'] = arr['signal'][0,0][7].ravel()
    # signal['target'] = arr['signal'][0,0][8].ravel()
    # # signal['coords'] = arr['signal'][0,0][9]
    # signal['ied'] = arr['signal'][0,0][10][0, 0]
    # signal['electrode'] = arr['signal'][0,0][5][0, 0]
    # signal['muscle'] = arr['signal'][0,0][6][0, 0][0]
    # # signal['nelectrodes'] = signal['nneedles'] + signal['ngrids']

    # # Load data for all grids (right now only keep second grid)
    # signal['data'] = arr['signal'][0,0][0]
    # pulse_trains = arr['edition'][0,0][0]
    # discharge_times = arr['edition'][0,0][1]

    # decomp_dict = {} # initialising this dictionary here for later use
    # mu_dict = dict(pulse_trains = None, discharge_times = [])# initialising a dictionary that is an empty nested list

    # return signal, pulse_trains, discharge_times, decomp_dict, mu_dict
    # return signal, edition

# def simon_grid_formatting(emg_obj):
#     ''' Takes grid, and based on the grid coordinate map, reorders the channels such that they can be made into a grid with the default reshaping settings.'''
#     ElChannelMap = [[0, 24, 25, 50, 51], 
#             [0, 23, 26, 49, 52], 
#             [1, 22, 27, 48, 53], 
#             [2, 21, 28, 47, 54], 
#             [3, 20, 29, 46, 55], 
#             [4, 19, 30, 45, 56], 
#             [5, 18, 31, 44, 57], 
#             [6, 17, 32, 43, 58],  
#             [7, 16, 33, 42, 59], 
#             [8, 15, 34, 41, 60],  
#             [9, 14, 35, 40, 61], 
#             [10, 13, 36, 39, 62], 
#             [11, 12, 37, 38, 63]]
    
#     # Remove channels from top row
#     channels2drop = [24, 25, 50, 51]
#     emg_obj.signal_dict['data'] = np.delete(emg_obj.signal_dict['data'], channels2drop, axis=0)

#     # Reorder channels based on snaking grid structure
#     grid = np.reshape(np.arange(60), (12, 5), order='F') # create initial indices such that indices increase first across rows, instead of default columns
#     for idx in range(1, grid.shape[1], 2):
#           grid[:, idx] = grid[::-1, idx] # flip every other column, then flatten array again to get reodering coordinats
#     reorder_idxs = grid.ravel()
#     emg_obj.signal_dict['data'] = emg_obj.signal_dict['data'][reorder_idxs, :]

#     # # CHECK THAT EMG image makes sense!
#     # import matplotlib.pyplot as plt
#     # rms = np.sqrt(emg_obj.signal_dict['data']**2).mean(axis=1)
#     # plt.figure()
#     # plt.imshow(rms.reshape(12, 5))
#     # plt.savefig('rms_img_final.jpg')

#     return emg_obj        

# def emanuele_grid_formatting(emg_obj):
#     ''' Takes grid, and based on the grid coordinate map, reorders the channels such that they can be made into a grid with the default reshaping settings.'''
#     ElChannelMap = np.array([[53,56,57,59,33,2,3,4,16,8,6,14,254,246,247,248,255,250,252,226,218,193,196,206],
#           [55,61,58,60,34,1,15,24,21,18,7,5,245,241,233,256,249,251,253,225,219,194,208,197],
#           [54,42,45,48,38,25,27,32,22,19,10,13,244,234,237,240,230,227,221,217,220,195,207,198],
#           [50,41,44,47,39,36,26,31,23,20,9,12,243,235,238,232,229,222,224,215,213,211,209,203],
#           [51,49,43,46,40,37,35,28,29,30,17,11,242,236,239,231,228,223,216,214,212,210,201,202],
#           [116,117,118,120,127,125,123,121,66,69,71,72,182,192,190,188,185,130,132,134,136,144,140,145],
#           [106,114,119,128,126,124,122,65,67,70,79,78,178,181,191,189,186,129,131,133,135,143,141,147],
#           [105,115,103,99,90,92,83,68,80,75,76,77,170,180,179,183,184,187,162,154,157,142,159,150],
#           [97,108,109,100,101,93,94,84,86,88,81,73,168,173,175,169,177,165,167,153,156,139,160,149],
#           [107,98,110,102,91,95,96,85,87,89,82,74,171,172,174,176,163,164,166,161,155,158,151,152]]).T-1

    # emg_obj.signal_dict['data'] = emg_obj.signal_dict['data'][ElChannelMap.ravel(), :] # should in priciple rearrange grid such that we can just use default reshape/ravel to get grid
    # # CHECK THAT EMG image makes sense!
    # import matplotlib.pyplot as plt
    # rms = np.std(emg_obj.signal_dict['data'], axis=1)
    # plt.figure()
    # plt.imshow(rms.reshape(24, 10))
    # plt.colorbar()
    # plt.savefig('rms_img_final.jpg')

    # return emg_obj

if __name__ == '__main__':
    DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/2mm'
    # DIR = '/home/joao/Desktop/datasets/emanuele_arnault/s1_edited/4mm'
    R = 8
    file = 'S1_25_2mm_Session1_MUEdit_edited.mat'
    print(os.listdir(DIR))
    signal, edition = open_mat_output(DIR, file)

    # Get data into desired shape
    emg = signal['data']
    emg_grid = emg[index_matrix, :]

    plt.figure()
    plt.imshow(emg_grid.std(axis=2))
    plt.show()

    # Get working separation vectors
    emg = emg - emg.mean(axis=1, keepdims=True) # centering emg
    extended_emg_template = np.zeros((R*emg.shape[0], emg.shape[1] + R - 1))
    extended_emg = extend_emg(extended_emg_template, emg, R)
    inv_cov = get_inv_cov(extended_emg)

    # Test motor unit discharge times
    dts = edition['Dischargetimes'][0,0].squeeze().astype(int)
    sep_vec = extended_emg[:, dts].mean(axis=1, keepdims=True)
    source_est = sep_vec.T @ inv_cov @ extended_emg
    constrast = source_est.squeeze()[20000:21000]**2
    constrast = (constrast - constrast.min()) / (constrast.max() - constrast.min())
    plt.figure()
    plt.plot(constrast)
    plt.plot(edition['Pulsetrain'][0,0][0,20000:21000])
    plt.show()


    # filenames = os.listdir(DIR)
    # for name in filenames:
    #     signal, edition = open_mat_output(DIR, name)
    #     # if ('4mm' not in name) and ('8mm' not in name):
    #     #     signal, edition = open_mat_output(DIR, name)
    #     #     fig, axs = plt.subplots(2,2)
    #     #     axs[0,0].imshow(signal['data'][:60].var(axis=1).reshape(12, 5))
    #     #     axs[0,1].imshow(signal['data'][64:124].var(axis=1).reshape(12, 5))
    #     #     axs[1,0].imshow(signal['data'][128:188].var(axis=1).reshape(12, 5))
    #     #     axs[1,1].imshow(signal['data'][192:252].var(axis=1).reshape(12, 5))
    #     #     plt.show()
    #     print(name, signal['data'].shape)

        # # for idx in range(arr['signal'][0,0][1].shape[1]):
        #     print('Removed electrodes:', np.where(arr['signal'][0,0][1][0,idx]))
        