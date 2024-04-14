import os
from time import time
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import  accuracy_score

# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, EMGFrameLoader
from utils.emg_processing import majority_voting_segments, train_model, test_model, b
from networks import CapgMyoNet, RMSNet

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    # Experiment condition loading
    name = 'capgmyo-rms-baseline-intrasession'
    with open('capgmyo.json') as f:
        data = json.load(f)
    with open('exp.json') as f:
        exp = json.load(f)
    # extract_frames = extract_frames_csl if data['dataset_name'] == 'csl' else extract_frames_capgmyo
    # data_extractor_def = CapgmyoData if data['dataset_name'] == 'capgmyo' else CSLData
    data_extractor_def = CapgmyoDataRMS

    t0 = time()

    # Preinitialize metric arrays
    subs, sessions = [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('Intrasession:', data['dataset_name'])
    for sub in tqdm(data['subs']):
        for ses in tqdm(data['sessions']):
            for test_idx in range(10):
                # Load data for given subject/session
                print('\n SUBJECT #{}, SESSION #{}'.format(sub, ses))
                sub_id = 'subject{}'.format(sub)
                session_id = 'session{}'.format(ses)
                subs.append(sub)
                sessions.append(ses)

                # Load EMG data in uniform format
                data_extractor = data_extractor_def(path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                                    input_shape=data['input_shape'], fs=data['fs'], sessions=session_id, median_filt=True, baseline=True)
                data_extractor.load_tensors()
                X_train, Y_train, X_test, Y_test = data_extractor.get_tensors(test_idx=9)
        
                # Get PyTorch DataLoaders
                train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
                test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

                # Model/training set-up
                model = CapgMyoNet(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_gestures']).to(device)
                num_epochs = exp['num_epochs']
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])

                # Train the model
                train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler) # run training loop

                # Testing loop over test loader
                model.eval()
                with torch.no_grad():
                    all_labs, all_preds = test_model(model, test_loader)

                acc = accuracy_score(all_labs, all_preds)
                accs.append(acc)
                print('Test Accuracy:', acc)

                # MAJORITY VOTING PREDICTIONS
                all_preds_maj = np.array(majority_voting_segments(all_preds, M=exp['M'], n_samples=data['fs']))
                maj_acc = accuracy_score(all_labs, all_preds_maj)
                maj_accs.append(maj_acc)
                print('Majority Voting Test Accuracy:', maj_acc)

                # SAVE RESULTS
                arr = np.array([subs, ses, accs, maj_accs]).T
                df = pd.DataFrame(data=arr, columns=['Subjects', 'Session', 'Test Rep' 'Accuracy', 'MV Accuracy'])
                df.to_csv('./{}.csv'.format(name))

    # Save experiment data in .csv file
    data = np.array([subs, ses, accs, maj_accs]).T
    df = pd.DataFrame(data=data, columns=['Subjects', 'Session', 'Test Rep', 'Accuracy', 'MV Accuracy'])
    df.to_csv('./{}.csv'.format(name))

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('TOTAL TIME ELAPSED: {}h, {}min'.format(h, m))
