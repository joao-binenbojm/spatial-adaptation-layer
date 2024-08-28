import matplotlib.pyplot as plt
import numpy as np
import json
import wandb
from networks_utils import median_pool_2d
from tqdm import tqdm
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS, CapgmyoDataSegmentRMS, CSLDataSegmentRMS

if __name__ == '__main__':

    # Conditions for obtaining data in images
    dataset = 'csl'
    emg_tensorizer_def = CSLDataSegmentRMS
    real_baseline = True
    Trms = 75
    with open('{}.json'.format(dataset)) as f:
        data = json.load(f)
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]

    for idx, sub in tqdm(enumerate(data['subs'])):
        # Load data for given subject/session
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        print('\nLOADING EMG TENSOR...')
        emg_tensorizer = emg_tensorizer_def(dataset=dataset, path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=session_ids, intrasession=False, Trms=Trms, remove_baseline=real_baseline)
        emg_tensorizer.load_tensors()

        # Run code 5 times for every train/test session pair, except where same session is used for train and test
        for session in data['sessions']:
                
            X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(
                                                                                    test_session=0,
                                                                                    train_session=session,
                                                                                    rep_idx=0)
            
            X_train_plot = median_pool_2d(X_train, kernel_size=(3,1), padding=(1,0))
            # X_train_plot = X_train
            for label in range(len(np.unique(Y_train))):
                Xmean = X_train_plot[Y_train == label, 0, :, :].mean(axis=0)
                plt.figure()
                plt.imshow(Xmean, cmap='gray')
                plt.savefig(f'mean_images/{dataset}/subject{sub+1}/session{session+1}/gest{label+1}.jpg')
                plt.close()