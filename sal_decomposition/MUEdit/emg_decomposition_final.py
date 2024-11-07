import glob, os, scipy
import xml.etree.ElementTree as ET
import tarfile as tf
import numpy as np
import matplotlib.pyplot as plt
from sal_decomposition.MUEdit.processing_tools import *
import tkinter as tk
from tkinter import simpledialog
from scipy import signal
import pickle
import time
# root = tk.Tk() # Initialising for GUI
np.random.seed(1337) # Fixes random generation to get same results each time the script is run

class EMG():

    def __init__(self):
        self.its = 120 # number of iterations of the fixed point algorithm 
        self.ref_exist = 1 # if ref_signal exist ref_exist = 1; if not ref_exist = 0 and manual selection of windows
        self.check_emg = 0 # 0 = Automatic selection of EMG channels (remove 5% of channels) ; 1 = Visual checking
        self.drawing_mode = 0 # 0 = Output in the command window ; 1 = Output in a figure
        self.differential_mode = 0 # 0 = no; 1 = yes (filter out the smallest MU, can improve decomposition at the highest intensities
        self.peel_off = 1 # 0 = no; 1 = yes (update the residual EMG by removing the motor units with the highest SIL value
        self.peel_off_win = 0.025 # if peel_off, the window around a spike to account for in seconds
        self.sil_thr = 0.9 # Threshold for SIL values when discarding MUs after two fastICA phases
        self.silthrpeeloff = 0.9 # Threshold for MU removed from the signal (if the sparse  deflation is on)
        self.ext_factor = 1000 # extension of observations for numerical stability 
        self.edges2remove = 0 # Extent of signal clipping after whitening 
        self.target_thres = 0.8  # Threshold for segmenting and batching the EMG signals based on a target force profile
        self.initialisation = 0 # 0 = initialisation based on the a maximum value in the EMG signal, 1 = random initialisaiton
        self.cov_thr = 0.5 # Threshold for CoV values when discarding MUs after two fastICA phases
        self.cov_filter = 1
        self.dup_thr = 0.3 # Correlation threshold for defining a pair of spike trains as derived from the same MU, hence a duplicate
        self.refine_mu = 1
        self.dup_bgrids = 0

#######################################################################################################
########################################## OFFLINE EMG ################################################
#######################################################################################################

class offline_EMG(EMG):

    # child class of EMG, so will inherit it's initialisaiton
    def __init__(self, save_dir, to_filter):
        super().__init__()
        self.save_dir = save_dir # directory at which final discharges will be saved
        self.to_filter = to_filter # whether or not you notch and butter filter the 
    
    def open_otb(self, inputfile):

        file_name = inputfile.split('/')[1]
        temp_dir = os.path.join(self.save_dir, 'temp_tarholder')

        # make a temporary directory to store the data of the otb file if it doesn't exist yet
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

        # Open the .tar file and extract all data
        with tf.open(inputfile, 'r') as emg_tar:
            emg_tar.extractall(temp_dir)

        #os.chdir(temp_dir)
        sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
        trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
        trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
        trial_label_sig = os.path.join(temp_dir, trial_label_sig)
        trial_label_xml = os.path.join(temp_dir, trial_label_xml)

        # read the contents of the trial xml file
        with open(trial_label_xml, encoding='utf-8') as file:
            xml=ET.fromstring(file.read())

        # get sampling frequency, no. bits of AD converter, no. channels, electrode names and muscle names
        fsamp = int(xml.find('.').attrib['SampleFrequency'])
        nADbit = int(xml.find('.').attrib['ad_bits'])
        nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
        electrode_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
        muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
        ngrids = sum(1 for name in electrode_names if name.startswith('GR'))
        nneedles = (sum(1 for name in electrode_names if any(substring.lower() in name.lower() for substring in ['aux', 'ramp', 'buffer'])) +  ngrids) - len(electrode_names)
        nelectrodes = nneedles + ngrids

        # read in the EMG trial data
        emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
        emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
        emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division

        # convert the data from bits to microvolts
        for i in range(nchans):
            emg_data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids, nneedles = nneedles, nelectrodes = nelectrodes, electrode = electrode_names[0],muscle = muscle_names[0]) # discard the other muscle and grid entries, not relevant

        # if the signals were recorded with a feedback generated by OTBiolab+, get the target and the path performed by the participant
        if self.ref_exist:

            # only opening the last two .sip files because the first is not needed for analysis
            # would only need MSE between the participant path (file 2) and the target path (file 3)
            ######## path #########
            _, target_label, path_label = glob.glob(f'{temp_dir}/*.sip')
            with open(path_label) as file:
                path = np.fromfile(file, dtype='float64')
                path = path[:np.shape(emg_data)[1]]
            ######## target ########
            with open(target_label) as file:
                target = np.fromfile(file,dtype='float64')
                target = target[:np.shape(emg_data)[1]]
            
            signal['path'] = path
            signal['target'] = target
        
        # delete the temp_tarholder directory since everything we need has been taken out of it
        for file_name in os.listdir(temp_dir):
            file = os.path.join(temp_dir, file_name)
            if os.path.isfile(file):
                os.remove(file)

        os.rmdir(temp_dir)

        self.signal_dict = signal
        self.decomp_dict = {} # initialising this dictionary here for later use
        self.mu_dict = dict(pulse_trains = None, discharge_times = []) # initialising a dictionary that is an empty nested list

    def segment_w_target(self):

        plateau = np.where(self.signal_dict['target'] >= max(self.signal_dict['target'])*self.target_thres)[0] # finding where the plateau is        
        # the last option is having only one window and no discontinuity in the plateau; in that case, you leave as is
        segment = [plateau[0],plateau[-1]]
        self.plateau_coords = segment
        
        # with the markers for windows and plateau discontinuities, batch the emg data ready for decomposition
        segmented_data = self.signal_dict['data'][:, int(self.plateau_coords[0]):int(self.plateau_coords[1])+1]
        self.signal_dict['segmented_data'] = segmented_data

        print('Data batched across all surface and intramusuclar arrays')
            
  
################################ CONVOLUTIVE SPHERING ########################################
    def convul_sphering(self):

        # g is the index for the electrodes
        """ 1) Filter the segmented EMG data 2) Extend to improve speed of convergence/reduce numerical instability 3) Remove any DC component  4) Whiten """

        print(self.emgopt)
        if self.to_filter: # adding since will need to avoid this step if doing real-time decomposition + biofeedback, but fine for offline analysis

            self.signal_dict['segmented_data'] = notch_filter(self.signal_dict['segmented_data'],self.signal_dict['fsamp'])
            self.signal_dict['segmented_data'] = bandpass_filter(self.signal_dict['segmented_data'],self.signal_dict['fsamp'],emg_type = self.emgopt)  

        # differentiation - typical EMG generation model treats low amplitude spikes/MUs as noise, which is common across channels so can be cancelled with a first order difference. Useful for high intensities - where cross talk has biggest impact.
        if self.differential_mode: # just a basic 1st order differential (bipolar processing)
           
            self.signal_dict['segmented_data'] = np.diff(self.signal_dict['segmented_data'],n=1,axis=-1)

        # signal extension - increasing the number of channels to 1000
        # Holobar 2007 -  Multichannel Blind Source Separation using Convolutive Kernel Compensation (describes matrix extension)
        extension_factor = int(np.round(self.ext_factor/len(self.signal_dict['segmented_data'])))
        self.ext_number =  extension_factor
        self.signal_dict['extend_obvs_old'] = extend_emg(self.signal_dict['extend_obvs_old'], self.signal_dict['segmented_data'], extension_factor)
        self.signal_dict['sq_extend_obvs'] = (self.signal_dict['extend_obvs_old'] @ self.signal_dict['extend_obvs_old'].T) / np.shape(self.signal_dict['extend_obvs_old'])[1]
        self.signal_dict['inv_extend_obvs'] = np.linalg.pinv(self.signal_dict['sq_extend_obvs']) # different method of pinv in MATLAB --> SVD vs QR
        
        # de-mean the extended emg observation matrix
        self.signal_dict['extend_obvs_old'] = scipy.signal.detrend(self.signal_dict['extend_obvs_old'], axis=- 1, type='constant', bp=0)
        
        # whiten the signal + impose whitened extended observation matrix has a covariance matrix equal to the identity for time lag zero
        self.decomp_dict['whitened_obvs_old'],self.decomp_dict['whiten_mat'], self.decomp_dict['dewhiten_mat'] = whiten_emg(self.signal_dict['extend_obvs_old'])
        
        # remove the edges
        self.signal_dict['extend_obvs'] = self.signal_dict['extend_obvs_old'][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)):-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        self.decomp_dict['whitened_obvs'] = self.decomp_dict['whitened_obvs_old'][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)):self.signal_dict['extend_obvs_old'].shape[1]-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        
        self.plateau_coords[0] = self.plateau_coords[0]  + int(np.round(self.signal_dict['fsamp']*self.edges2remove))
        self.plateau_coords[1] = self.plateau_coords[1]  - int(np.round(self.signal_dict['fsamp']*self.edges2remove))

        print('Signal extension and whitening complete')
        
######################### FAST ICA AND CONVOLUTIVE KERNEL COMPENSATION  ############################################

    def fast_ICA_and_CKC(self, cf_type = 'square'):

        
        init_its = np.zeros([self.its],dtype=int) # tracker of initialisaitons of separation vectors across iterations
        fpa_its = 500 # maximum number of iterations for the fixed point algorithm
       
       
        ####### TESTING WITH A RANDOM (SEEDED) DATA MATRIX ###########

        # random test to compare to matlab
        # self.decomp_dict['whitened_obvs'] = np.random.random((np.shape(self.decomp_dict['whitened_obvs'])[1],np.shape(self.decomp_dict['whitened_obvs'])[0])).T

        Z = np.array(self.decomp_dict['whitened_obvs']).copy() # copy of the whitened signal, that will be modified through fast ICA
        raw_sources = []
        time_axis = np.linspace(0,np.shape(Z)[1],np.shape(Z)[1])/self.signal_dict['fsamp']  # create a time axis for spiking activity

        # choosing contrast function here, avoid repetitively choosing within the iteration loop
        if cf_type == 'square':
            cf = square
            dot_cf = dot_square
        elif cf_type == 'skew':
            cf = skew
            dot_cf = dot_skew
        elif cf_type == 'exp':
            cf = exp
            dot_cf = dot_exp
        elif cf_type == 'logcosh':
            cf = logcosh
            dot_cf = dot_logcosh
      
        for i in range(self.its):

                #################### FIXED POINT ALGORITHM #################################

                if self.initialisation:
                    # generate a random vector
                    random_init = np.random.randn(np.size(self.decomp_dict['whitened_obvs'])[0],np.size(self.decomp_dict['whitened_obvs'])[0]) # dimension extended channels x extended channels
                    self.decomp_dict['w_sep_vect'] = random_init[:,0]
                else:
                    if i == 0 :
                        # identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
                         # occurs. Then, the projection vector is initialised to the whitened observation vector, at this located time instant.
                        sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 0)))

                    init_its[i] = sort_sq_sum_Z[-(i+1)] # since the indexing starts at -1 the other way (for ascending order list)
                    self.decomp_dict['w_sep_vect'] = Z[:,int(init_its[i])].copy() # retrieve the corresponding signal value to initialise the separation vector
                
                # orthogonalise separation vector before fixed point algorithm
                self.decomp_dict['w_sep_vect'] -= np.dot(self.decomp_dict['B_sep_mat'] @ self.decomp_dict['B_sep_mat'].T, self.decomp_dict['w_sep_vect'])
               
                # normalise separation vector before fixed point algorithm 
                self.decomp_dict['w_sep_vect'] /= np.linalg.norm(self.decomp_dict['w_sep_vect'])
            
                # use the fixed point algorithm to identify consecutive separation vectors
                self.decomp_dict['w_sep_vect'] = fixed_point_alg(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'],Z, cf, dot_cf,fpa_its)
                
                # get the first iteration of spikes using k means ++
                fICA_source, spikes = get_spikes(self.decomp_dict['w_sep_vect'],Z, self.signal_dict['fsamp'])
                raw_sources.append(fICA_source)
            
                ################# MINIMISATION OF COV OF DISCHARGES ############################
                if len(spikes) > 1:

                    # determine the interspike interval
                    ISI = np.diff(spikes/self.signal_dict['fsamp'])
                    # determine the coefficient of variation
                    CoV = np.std(ISI)/np.mean(ISI)
                    # update the sepearation vector by summing all the spikes
                    w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
                    # minimisation of covariance of interspike intervals
                    self.decomp_dict['MU_filters'][:,i], spikes, self.decomp_dict['CoVs'][i] = min_cov_isi(w_n_p1,self.decomp_dict['B_sep_mat'],Z, self.signal_dict['fsamp'],CoV,spikes)
                    self.decomp_dict['B_sep_mat'][:,i] = (self.decomp_dict['w_sep_vect']).real # no need to shallow copy here

                    # calculate SIL
                    fICA_source, spikes, self.decomp_dict['SILs'][i] = get_silohuette(self.decomp_dict['MU_filters'][:,i],Z,self.signal_dict['fsamp'])
                    # peel off
                    if self.peel_off == 1 and self.decomp_dict['SILs'][i] > self.sil_thr:
                        Z = peel_off(Z, spikes, self.signal_dict['fsamp'], peel_off_win=self.peel_off_win)


                    print(self.decomp_dict['SILs'][i])
            
                    if self.drawing_mode == 1:
                        plt.clf()
                        plt.ion()
                        plt.show()
                        plt.subplot(2, 1, 1)
                        plt.plot(self.signal_dict['target'], 'k--', linewidth=2)
                        plt.plot([self.plateau_coords[0], self.plateau_coords[0]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.plot([self.plateau_coords[1], self.plateau_coords[1]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.title('Electrode #{} - Iteration #{} - Sil = {}'.format(g, i+1, self.decomp_dict['SILs'][i]))
                        plt.subplot(2, 1, 2)
                        plt.plot(time_axis, fICA_source,linewidth = 0.5)
                        plt.plot(time_axis[spikes],fICA_source[spikes],'o')
                        plt.grid()
                        plt.draw()
                        plt.pause(1e-6)
                    else:
                        print('Iteration #{} - Sil = {} - CoV = {}'.format(i, self.decomp_dict['SILs'][i],self.decomp_dict['CoVs'][i]))

                else:
                    print('Iteration #{} - less than 10 spikes '.format(i))
                    # without enough spikes, we skip minimising the covariation of discharges to improve the separation vector
                    self.decomp_dict['B_sep_mat'][:,i] = self.decomp_dict['w_sep_vect'].real  # no need to shallow copy here
                self.mu_dict['raw_sources'] = np.array(raw_sources)

        ####################################### MU FILTER THRESHOLDING ###############################################

        # remove the MU filters that fall below the imposed metric thresholds
        SIL_condition = self.decomp_dict['SILs'] >= self.sil_thr
        final_condition = SIL_condition.copy()

        if self.cov_filter:
            # mask combines CoV and SIL threshold crtieria
            CoV_condition = self.decomp_dict['CoVs'] <= self.cov_thr
            final_condition = SIL_condition & CoV_condition


        mask = np.broadcast_to(final_condition.reshape(1, -1), (np.shape(self.decomp_dict['whitened_obvs'])[0], self.its))

        self.decomp_dict['masked_mu_filters'] = self.decomp_dict['MU_filters'][mask].reshape(np.shape(self.decomp_dict['whitened_obvs'])[0], np.sum(mask, axis=1)[0])
        self.decomp_dict['sources'] = fICA_source

    
        plt.close() #closes the fixed point algorithm plots
        
################################################## POST PROCESSING #######################################################

    def post_process_EMG(self):

        # self.mus_in_array = np.zeros(self.signal_dict['nelectrodes'])
        # batch processing over each window
        pulse_trains, discharge_times = batch_process_filters(self.decomp_dict['whitened_obvs'],self.decomp_dict['masked_mu_filters'],self.plateau_coords,self.ext_number,self.differential_mode,np.shape(self.signal_dict['data'])[1],self.signal_dict['fsamp'])

        if np.shape(pulse_trains)[0] > 0: # if there are existing MUs
            
            # self.mus_in_array[electrode-1] = 1 # if pulse trains were not extracted for this electrode, then it remains at a value of 0
            # removing duplicate MUs
            discharge_times_new, pulse_trains_new, mu_filters_new =  remove_duplicates(pulse_trains, discharge_times,discharge_times,np.squeeze(self.decomp_dict['masked_mu_filters']),np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])
            self.decomp_dict['masked_mu_filters'] = []
            self.decomp_dict['masked_mu_filters'] = mu_filters_new

            if self.refine_mu:
                # removing outliers generating irrelvant discharge rates before manual edition (1st time)
                discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'], self.cov_thr)
                
                pulse_trains_new, discharge_times_new = refine_mus(self.signal_dict['data'], pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'])

                discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'], self.cov_thr)
            
            self.mu_dict['pulse_trains'] = pulse_trains_new
            
            # self.mu_dict['discharge_times'].append([])

        for j in range(len(discharge_times_new)):

            self.mu_dict['discharge_times'].append(discharge_times_new[j])

        
    