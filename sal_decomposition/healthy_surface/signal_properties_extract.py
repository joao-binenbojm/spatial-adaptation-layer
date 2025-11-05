import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sal_decomposition.utils import utils
from sal_decomposition.utils.grid_indexing import index_matrix4, index_matrix2


if __name__ == '__main__':

    # Parameters
    ied = 4
    H, W = 26, 10
    index_matrix = index_matrix4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsamp = 2048

    data_dict = {'subject': [], 'session': [], 'mvc': [], 'noutliers': [], 'median_spatial_coherence': []}

    for idx, subject in enumerate([0,1,2]):
        DIR = f'/home/joao/Desktop/datasets/emanuele_arnault/s{subject+1}_edited'
        if subject == 0:
            DIR  = os.path.join(DIR, f'{ied}mm')

        for mvc in [25,50]:
            for ses in [0,1,2]:
                    
                # Load both sessions from the same subject
                file = f'S{subject+1}_{mvc}_Session{ses+1}_MUEdit_edited.mat'

                # Load training data
                sgnl, edition = utils.open_mat_output(DIR, file)
                start, end = utils.get_target_boundaries(sgnl['target'].squeeze())
                print('FILTERING TRAINING DATA...')
                emg = sgnl['data']
                emg = emg - emg.mean(axis=0, keepdims=True) # average referencing
                emg = utils.bandstop_filter(utils.bandpass_filter(emg, fsamp=fsamp), fsamp=fsamp)
                emg = (emg - emg.mean(axis=1, keepdims=True)) / (emg.std(axis=1, keepdims=True) + 1e-12) # centering emg
                emg_grid_original = utils.make_grid(emg, index_matrix4)
                
                H, W = emg_grid_original.shape[2], emg_grid_original.shape[3]
                Nch = H*W

                # Compute outliers as channels average of neighbours
                print('COUNTING OUTLIER CHANNELS...')
                emg_grid = emg_grid_original.numpy()
                flatness = utils.get_spectral_flatness_ar2(emg_grid_original.squeeze())
                Q1, Q3 = np.quantile(flatness.flatten(), [0.25, 0.75])
                IQR = Q3 - Q1
                upper = Q3 + 1.5*IQR
                outlier_mask = flatness >= upper  # boolean mask of outliers
                noutliers = np.sum(outlier_mask)

                # Get median spatial coherence
                spatial_coherence = utils.calculate_spatial_coherence(emg_grid_original)
                median_spatial_coherence = np.median(spatial_coherence)
                print(f'Subject {subject+1}, MVC {mvc}%, Session {ses+1}: {noutliers} outlier channels detected. Median spatial coherence: {median_spatial_coherence:.4f}')

                # Store results
                data_dict['subject'].append(subject+1)
                data_dict['session'].append(ses+1)
                data_dict['mvc'].append(mvc)
                data_dict['noutliers'].append(int(noutliers))
                data_dict['median_spatial_coherence'].append(float(median_spatial_coherence))
    
    # Save results to a CSV file
    df = pd.DataFrame(data_dict)
    df['Subject & Session'] = df['subject'].astype(str) + ' - ' + df['session'].astype(str)
    df.to_csv('signal_properties_summary.csv', index=False)

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    sns.barplot(data=df, x='Subject & Session', y='noutliers', hue='mvc', ax=axs[0])
    axs[0].set_title('Number of Outlier Channels')
    sns.barplot(data=df, x='Subject & Session', y='median_spatial_coherence', hue='mvc', ax=axs[1])
    axs[1].set_title('Median Spatial Coherence')
    axs[1].set_ylim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig('signal_properties_summary.png', dpi=300)
    plt.show()

