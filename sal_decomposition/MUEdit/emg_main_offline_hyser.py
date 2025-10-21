from emg_decomposition_final import EMG, offline_EMG
import glob, os
import numpy as np
import pickle 
import pandas as pd
import wfdb
import json
from scipy.io import loadmat
import scipy

from sal_decomposition.utils import utils
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def open_mat_output(DIR, subject=1, session=1, finger=1, sample=1, task="flexion"):
    ''' Open hyser data in the desired format for Ciara's decomposition script/code.'''
    # arr = loadmat(os.path.join(DIR, name))
    subses_dir = os.path.join(DIR, f"subject0{subject}_session{session}")
    record_emg = wfdb.rdrecord(os.path.join(subses_dir, f"1dof_preprocess_finger{finger}_sample{sample}"))
    record_force = wfdb.rdrecord(os.path.join(subses_dir, f"1dof_force_finger{finger}_sample{sample}"))
    # grid_idx = 1
    
    # Reshaping of the grid
    ngrids = 4
    emg = record_emg.p_signal
    subimages = []
    for grid_idx in range(ngrids):
        subimage = np.array(emg[:, grid_idx*64:(grid_idx+1)*64]).reshape(emg.shape[0], 1, 8, 8)
        subimage = np.flip(np.flip(subimage, axis=2), axis=3)
        subimages.append(subimage)
    images = np.concatenate(subimages, axis=2) # stack grids to get final image

    if task == "extension":
        visible_outliers = np.zeros((8*2, 8), dtype=np.bool_)
        images = utils.handle_outliers(torch.tensor(images[:, :, :16, :]), visible_outliers=visible_outliers).numpy()
        emg = images.squeeze().reshape(images.shape[0], 8*8*2)
    elif task == "flexion":
        visible_outliers = np.zeros((8*2, 8), dtype=np.bool_)
        visible_outliers[4, 2] = True
        images = utils.handle_outliers(torch.tensor(images[:, :, 16:, :]), visible_outliers=visible_outliers).numpy()
        emg = images.squeeze().reshape(images.shape[0], 8*8*2)
    else:
        raise Exception("Incorrect task requested.")

    # Get target force for specific desired finger, then upsample to 2048Hz
    target = scipy.signal.resample(record_force.p_signal[:, finger-1], num=record_emg.p_signal.shape[0])

    # Load acquisition signal
    signal = {}
    signal['nchans'] = 128 # 128
    signal['fsamp'] = record_emg.fs
    signal['path'] = target
    signal['target'] = target
    # signal['coords'] = arr['signal'][0,0][9]
    signal['ied'] = 10

    # Load data for all grids (right now only keep second grid)
    signal['data'] = emg
    pulse_trains = None

    decomp_dict = {} # initialising this dictionary here for later use
    mu_dict = dict(pulse_trains = None, discharge_times = [])# initialising a dictionary that is an empty nested list

    return signal, pulse_trains, decomp_dict, mu_dict

# def grid_formatting(emg_obj):
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
    
    # Remove channels from top row
    # channels2drop = [24, 25, 50, 51]
    # emg_obj.signal_dict['data'] = np.delete(emg_obj.signal_dict['data'], channels2drop, axis=0)

    # Reorder channels based on snaking grid structure
    grid = np.reshape(np.arange(60), (12, 5), order='F') # create initial indices such that indices increase first across rows, instead of default columns
    for idx in range(1, grid.shape[1], 2):
          grid[:, idx] = grid[::-1, idx] # flip every other column, then flatten array again to get reodering coordinats
    reorder_idxs = grid.ravel()
    emg_obj.signal_dict['data'] = emg_obj.signal_dict['data'][reorder_idxs, :]

    # # CHECK THAT EMG image makes sense!
    # import matplotlib.pyplot as plt
    # rms = np.sqrt(emg_obj.signal_dict['data']**2).mean(axis=1)
    # plt.figure()
    # plt.imshow(rms.reshape(12, 5))
    # plt.savefig('rms_img_final.jpg')

def grid_formatting(emg_obj):
    ngrids = 2
    emg = emg_obj.signal_dict['data']
    subimages = []
    for grid_idx in range(ngrids):
        subimage = np.array(emg[:, grid_idx*64:(grid_idx+1)*64]).reshape(emg.shape[0], 1, 8, 8)
        subimage = np.flip(np.flip(subimage, axis=2), axis=3)
        subimages.append(subimage)
    images = np.concatenate(subimages, axis=2) # stack grids to get final image
    emg = images.squeeze().reshape(images.shape[0], 8*8*2)
    emg_obj.signal_dict['data']

    return emg_obj

DIR = '/home/joao/Desktop/datasets/hyser/physionet.org/files/hd-semg/1.0.0/1dof_dataset/'
emg_obj = offline_EMG('./results',1)
emg_obj.to_filter = 0

########### Update: 12/03/24 - adding different file options for opening...
    
# Load mat file with signals
signal, pulse_trains, decomp_dict, mu_dict = open_mat_output(DIR, subject=2, session=2, finger=2, sample=1, task="extension")
emg_obj.signal_dict = signal
emg_obj.decomp_dict = decomp_dict
emg_obj.mu_dict = mu_dict
# emg_obj = grid_formatting(emg_obj)
emg_obj.signal_dict['data'] = emg_obj.signal_dict['data'].T # reorders signals such that a default reshape makes our rectangular grid
emg_obj.emgopt = 'surface'
print('Target used for segmenting')
# emg_obj.segment_w_target()
emg_obj.signal_dict['segmented_data'] = emg_obj.signal_dict['data']
emg_obj.plateau_coords = [0, emg_obj.signal_dict['data'].shape[1]]


################### CONVOLUTIVE SPHERING #############################

emg_obj.signal_dict['diff_data'] = []
tracker = 0
# nwins = int(len(emg_obj.plateau_coords)/2)
        
extension_factor = int(np.round(emg_obj.ext_factor/np.shape(emg_obj.signal_dict['segmented_data'])[0]))
# these two arrays are holding extended emg data PRIOR to the removal of edges
emg_obj.signal_dict['extend_obvs_old'] = np.zeros([np.shape(emg_obj.signal_dict['segmented_data'])[0]*(extension_factor), np.shape(emg_obj.signal_dict['segmented_data'])[1] + extension_factor -1 - emg_obj.differential_mode ])
emg_obj.decomp_dict['whitened_obvs_old'] = emg_obj.signal_dict['extend_obvs_old'].copy()
# # these two arrays are the square and inverse of extended emg data PRIOR to the removal of edges
emg_obj.signal_dict['sq_extend_obvs'] = np.zeros([np.shape(emg_obj.signal_dict['segmented_data'])[0]*(extension_factor),np.shape(emg_obj.signal_dict['segmented_data'])[0]*(extension_factor)])
emg_obj.signal_dict['inv_extend_obvs'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
# # dewhitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
emg_obj.decomp_dict['dewhiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
# whitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
emg_obj.decomp_dict['whiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
# these two arrays are holding extended emg data AFTER the removal of edges
emg_obj.signal_dict['extend_obvs'] = emg_obj.signal_dict['extend_obvs_old'][:,int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove)):emg_obj.signal_dict['extend_obvs_old'].shape[1]-int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove))].copy()
emg_obj.decomp_dict['whitened_obvs'] = emg_obj.signal_dict['extend_obvs'].copy()
    
# initialise zero arrays for separation matrix B and separation vectors w
emg_obj.decomp_dict['B_sep_mat'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'])[0],emg_obj.its])
emg_obj.decomp_dict['w_sep_vect'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'])[0],1])
emg_obj.decomp_dict['MU_filters'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'])[0],emg_obj.its])
emg_obj.decomp_dict['SILs'] = np.zeros([emg_obj.its])
emg_obj.decomp_dict['CoVs'] = np.zeros([emg_obj.its])
emg_obj.decomp_dict['tracker'] =  np.zeros([1,emg_obj.its])
emg_obj.decomp_dict['masked_mu_filters'] = []   # initialise empty list for the MU filters, because at each interval the removed MUs might not be the same
    
emg_obj.convul_sphering()

#################### FAST ICA ########################################
emg_obj.fast_ICA_and_CKC()

tracker = tracker + 1

##################### POSTPROCESSING #################################

emg_obj.post_process_EMG()

emg_grid = np.transpose(emg_obj.signal_dict['data']).reshape(emg_obj.signal_dict['data'].shape[1], 1, 16, 8)

####################### SAVING DECOMPOSITION OUTPUT INTO DATAFRAMES ########################
minimal_dict = {"emg_grid": emg_grid, 
                "target": emg_obj.signal_dict['target'], 
                "discharge_times": emg_obj.mu_dict['discharge_times']}

with open('./sal_decomposition/MUEdit/decomposition_data.pkl','wb') as file:
        pickle.dump(minimal_dict, file)

print()

# # Creating a decomposition dictionary as output
# parameters_dict = {
#     'file_path': filename, 'n_its':  emg_obj.its, 'ref_exist': emg_obj.ref_exist, 'fsamp': emg_obj.signal_dict['fsamp'],
#     'target_thres': emg_obj.target_thres, 'nwins': nwins, 'check_EMG': emg_obj.check_emg,
#     'drawing_mode': emg_obj.drawing_mode, 'differential_mode': emg_obj.differential_mode, 'peel_off': emg_obj.peel_off,
#     'refine_MU': emg_obj.refine_mu, 'sil_thr': emg_obj.sil_thr, 'ext_factor': emg_obj.ext_factor,
#     'edges': emg_obj.edges2remove, 'dup_thr': emg_obj.dup_thr, 'cov_filter': emg_obj.cov_filter, 
#     'cov_thr': emg_obj.cov_thr}


# decomposition_dict = {'chan_name': emg_obj.signal_dict['electrode'],'muscle_name': emg_obj.signal_dict['muscle'],
#                     'ied': emg_obj.signal_dict['ied'], 'mu_filters': emg_obj.decomp_dict['masked_mu_filters'], 'whiten_mat': emg_obj.decomp_dict['whiten_mat'],
#                     'inv_extend_obvs': emg_obj.signal_dict['inv_extend_obvs'],'pulse_trains': [emg_obj.mu_dict['pulse_trains']],
#                     'discharge_times': emg_obj.mu_dict['discharge_times'], 'original_data': emg_obj.signal_dict['data'], 'filtered_data': emg_obj.signal_dict['segmented_data'],
#                     'raw_sources': emg_obj.mu_dict['raw_sources'],
#                     'path': emg_obj.signal_dict['path'], 'target': emg_obj.signal_dict['target'], 'plateau': emg_obj.plateau_coords, 'parameters': parameters_dict}

# # save dicitionaries into data file
# with open('./sal_decomposition/decomposition_data.pkl','wb') as file:
#         pickle.dump(decomposition_dict, file)

# print('Decomposed data saved.')


            
            


