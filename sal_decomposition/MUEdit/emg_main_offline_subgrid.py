## ONLY DECOMPOSE A CENTRAL SUBGRID FROM THE ORIGINAL MAJOR GRID ##
from emg_decomposition_final import EMG, offline_EMG
import glob, os
import numpy as np
import pickle 
import pandas as pd
import json
from scipy.io import loadmat
from sal_decomposition.MUEdit.processing_tools import bandpass_filter, notch_filter

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

def simon_grid_formatting(emg_obj):
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

def emanuele_grid_formatting(emg_obj):
    ''' Takes grid, and based on the grid coordinate map, reorders the channels such that they can be made into a grid with the default reshaping settings.'''
    ElChannelMap = np.array([[53,56,57,59,33,2,3,4,16,8,6,14,254,246,247,248,255,250,252,226,218,193,196,206],
          [55,61,58,60,34,1,15,24,21,18,7,5,245,241,233,256,249,251,253,225,219,194,208,197],
          [54,42,45,48,38,25,27,32,22,19,10,13,244,234,237,240,230,227,221,217,220,195,207,198],
          [50,41,44,47,39,36,26,31,23,20,9,12,243,235,238,232,229,222,224,215,213,211,209,203],
          [51,49,43,46,40,37,35,28,29,30,17,11,242,236,239,231,228,223,216,214,212,210,201,202],
          [116,117,118,120,127,125,123,121,66,69,71,72,182,192,190,188,185,130,132,134,136,144,140,145],
          [106,114,119,128,126,124,122,65,67,70,79,78,178,181,191,189,186,129,131,133,135,143,141,147],
          [105,115,103,99,90,92,83,68,80,75,76,77,170,180,179,183,184,187,162,154,157,142,159,150],
          [97,108,109,100,101,93,94,84,86,88,81,73,168,173,175,169,177,165,167,153,156,139,160,149],
          [107,98,110,102,91,95,96,85,87,89,82,74,171,172,174,176,163,164,166,161,155,158,151,152]]).T-1

    emg_obj.signal_dict['data'] = emg_obj.signal_dict['data'][ElChannelMap.ravel(), :] # should in priciple rearrange grid such that we can just use default reshape/ravel to get grid
    # CHECK THAT EMG image makes sense!
    import matplotlib.pyplot as plt
    rms = np.std(emg_obj.signal_dict['data'], axis=1)
    plt.figure()
    plt.imshow(rms.reshape(24, 10))
    plt.colorbar()
    plt.savefig('rms_img_final.jpg')

    return emg_obj        

def grid_crop(emg_obj, xcrop: int, ycrop: int, grid_shape):
    ''' Keep only a subgrid at the center, returning a signal of shape (H - 2ycrop, W - 2xcrop)'''
    emg_obj.signal_dict['uncropped_data'] = emg_obj.signal_dict['segmented_data'].copy()
    # filter signal as it will be filtered later
    emg_obj.signal_dict['uncropped_data'] = notch_filter(emg_obj.signal_dict['uncropped_data'],emg_obj.signal_dict['fsamp'])
    emg_obj.signal_dict['uncropped_data'] = bandpass_filter(emg_obj.signal_dict['uncropped_data'],emg_obj.signal_dict['fsamp'],emg_type = emg_obj.emgopt)  

    emg_grid = emg_obj.signal_dict['segmented_data'].reshape(grid_shape + (emg_obj.signal_dict['segmented_data'].shape[1],))
    emg_obj.signal_dict['segmented_data'] = emg_grid[ycrop:emg_grid.shape[0]-ycrop, xcrop:emg_grid.shape[1]-xcrop, :].reshape(-1, emg_grid.shape[2])


# DIR = '/home/joao/Desktop/datasets/simon/ta_grid'
# filename = 'S1_20_DF.otb+_decomp.mat_edited.mat'
DIR = '/home/joao/Desktop/datasets/DatasetEmanuele/Subject3'
filename = 'S_3_50_1.otb+'
emg_obj = offline_EMG('./results',1)
ycrop, xcrop = 2, 2
grid_shape = (24, 10) # (12, 5)
filetype = 'otb'

########### Update: 12/03/24 - adding different file options for opening...

# Load mat file with signals
if filetype == 'mat':
    signal, pulse_trains, decomp_dict, mu_dict = open_mat_output(DIR, filename)
    emg_obj.signal_dict = signal
    emg_obj.decomp_dict = decomp_dict
    emg_obj.mu_dict = mu_dict
    emg_obj = simon_grid_formatting(emg_obj) # reorders signals such that a default reshape makes our rectangular grid
    emg_obj.emgopt = 'surface'

elif filetype == 'otb':
    emg_obj.open_otb(os.path.join(DIR, filename))
    emg_obj = emanuele_grid_formatting(emg_obj) # reorders signals such that a default reshape makes our rectangular grid
    emg_obj.emgopt = 'surface'
    emg_obj.signal_dict['ied'] = 4

#################################### SEGMENTING ############################################
print('Target used for segmenting')
emg_obj.segment_w_target()
grid_crop(emg_obj, xcrop=xcrop, ycrop=ycrop, grid_shape=grid_shape)

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
emg_obj.fast_ICA_and_CKC(cf_type='logcosh')

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
    'cov_thr': emg_obj.cov_thr, 'xcrop': xcrop, 'ycrop': ycrop}


decomposition_dict = {'chan_name': emg_obj.signal_dict['electrode'],'muscle_name': emg_obj.signal_dict['muscle'],
                    'ied': emg_obj.signal_dict['ied'], 'mu_filters': emg_obj.decomp_dict['masked_mu_filters'], 'whiten_mat': emg_obj.decomp_dict['whiten_mat'],
                    'inv_extend_obvs': emg_obj.signal_dict['inv_extend_obvs'],'pulse_trains': [emg_obj.mu_dict['pulse_trains']],
                    'discharge_times': emg_obj.mu_dict['discharge_times'], 'original_data': emg_obj.signal_dict['data'], 'filtered_data': emg_obj.signal_dict['segmented_data'],
                    'raw_sources': emg_obj.mu_dict['raw_sources'], 'uncropped_data': emg_obj.signal_dict['uncropped_data'],
                    'path': emg_obj.signal_dict['path'], 'target': emg_obj.signal_dict['target'], 'plateau': emg_obj.plateau_coords, 'parameters': parameters_dict}

# save dicitionaries into data file
with open('./sal_decomposition/decomposition_data.pkl','wb') as file:
        pickle.dump(decomposition_dict, file)

print('Decomposed data saved.')


            
            


