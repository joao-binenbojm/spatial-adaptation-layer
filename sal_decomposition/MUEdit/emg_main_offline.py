from emg_decomposition_final import EMG, offline_EMG
import glob, os
import numpy as np
import pickle 
import pandas as pd
import json
from scipy.io import loadmat

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def open_mat_output(DIR, name, temp_dir='./temp', MVC=20):
    ''' Open data from Simon's files so that it can be decomposed with Ciara's code.'''
    arr = loadmat(os.path.join(DIR, name))
    grid_idx = 1

    # Load acquisition signal
    signal = {}
    signal['nchans'] = 64
    signal['fsamp'] = arr['signal'][0,0][2][0,0]
    # signal['ngrids'] = arr['signal'][0,0][4][0,0]
    signal['path'] = arr['signal'][0,0][7].ravel()
    signal['target'] = arr['signal'][0,0][8].ravel()
    # signal['coords'] = arr['signal'][0,0][9]
    signal['ied'] = arr['signal'][0,0][10][0, grid_idx]
    signal['electrode'] = arr['signal'][0,0][5][0, grid_idx][0]
    signal['muscle'] = arr['signal'][0,0][6][0, grid_idx][0]
    # signal['nelectrodes'] = signal['nneedles'] + signal['ngrids']

    # Load data for all grids (right now only keep second grid)
    signal['data'] = arr['signal'][0,0][0][(grid_idx)*signal['nchans']:(grid_idx + 1)*signal['nchans'],:]
    pulse_trains = arr['signal'][0,0][13][0,1]

    decomp_dict = {} # initialising this dictionary here for later use
    mu_dict = dict(pulse_trains = None, discharge_times = [])# initialising a dictionary that is an empty nested list

    return signal, pulse_trains, decomp_dict, mu_dict

def grid_formatting(emg_obj):
    ''' Takes grid, and based on the grid coordinate map, reorders the channels such that they can be made into a grid with the default reshaping settings.'''
    ElChannelMap = [[0, 24, 25, 50, 51], 
            [0, 23, 26, 49, 52], 
            [1, 22, 27, 48, 53], 
            [2, 21, 28, 47, 54], 
            [3, 20, 29, 46, 55], 
            [4, 19, 30, 45, 56], 
            [5, 18, 31, 44, 57], 
            [6, 17, 32, 43, 58],  
            [7, 16, 33, 42, 59], 
            [8, 15, 34, 41, 60],  
            [9, 14, 35, 40, 61], 
            [10, 13, 36, 39, 62], 
            [11, 12, 37, 38, 63]]
    
    # Remove channels from top row
    channels2drop = [24, 25, 50, 51]
    emg_obj.signal_dict['data'] = np.delete(emg_obj.signal_dict['data'], channels2drop, axis=0)

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

    return emg_obj        


DIR = '/home/joao/Desktop/datasets/simon/ta_grid'
filename = 'S1_20_DF.otb+_decomp.mat_edited.mat'
emg_obj = offline_EMG('./results',1)

########### Update: 12/03/24 - adding different file options for opening...
    
# Load mat file with signals
signal, pulse_trains, decomp_dict, mu_dict = open_mat_output(DIR, filename)
emg_obj.signal_dict = signal
emg_obj.decomp_dict = decomp_dict
emg_obj.mu_dict = mu_dict
emg_obj = grid_formatting(emg_obj) # reorders signals such that a default reshape makes our rectangular grid
emg_obj.emgopt = 'surface'
print('Target used for segmenting')
emg_obj.segment_w_target()

################### CONVOLUTIVE SPHERING #############################

emg_obj.signal_dict['diff_data'] = []
tracker = 0
nwins = int(len(emg_obj.plateau_coords)/2)
        
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

####################### SAVING DECOMPOSITION OUTPUT INTO DATAFRAMES ########################

# Creating a decomposition dictionary as output
parameters_dict = {
    'file_path': filename, 'n_its':  emg_obj.its, 'ref_exist': emg_obj.ref_exist, 'fsamp': emg_obj.signal_dict['fsamp'],
    'target_thres': emg_obj.target_thres, 'nwins': nwins, 'check_EMG': emg_obj.check_emg,
    'drawing_mode': emg_obj.drawing_mode, 'differential_mode': emg_obj.differential_mode, 'peel_off': emg_obj.peel_off,
    'refine_MU': emg_obj.refine_mu, 'sil_thr': emg_obj.sil_thr, 'ext_factor': emg_obj.ext_factor,
    'edges': emg_obj.edges2remove, 'dup_thr': emg_obj.dup_thr, 'cov_filter': emg_obj.cov_filter, 
    'cov_thr': emg_obj.cov_thr}


decomposition_dict = {'chan_name': emg_obj.signal_dict['electrode'],'muscle_name': emg_obj.signal_dict['muscle'],
                    'ied': emg_obj.signal_dict['ied'], 'mu_filters': emg_obj.decomp_dict['masked_mu_filters'], 'whiten_mat': emg_obj.decomp_dict['whiten_mat'],
                    'inv_extend_obvs': emg_obj.signal_dict['inv_extend_obvs'],'pulse_trains': [emg_obj.mu_dict['pulse_trains']],
                    'discharge_times': emg_obj.mu_dict['discharge_times'], 'original_data': emg_obj.signal_dict['data'], 'filtered_data': emg_obj.signal_dict['segmented_data'],
                    'raw_sources': emg_obj.mu_dict['raw_sources'],
                    'path': emg_obj.signal_dict['path'], 'target': emg_obj.signal_dict['target'], 'plateau': emg_obj.plateau_coords, 'parameters': parameters_dict}

# save dicitionaries into data file
with open('./sal_decomposition/decomposition_data.pkl','wb') as file:
        pickle.dump(decomposition_dict, file)

print('Decomposed data saved.')


            
            


