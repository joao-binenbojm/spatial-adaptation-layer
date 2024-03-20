import os
import sys

import scipy
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
from skorch import NeuralNetClassifier

import preprocess_functions as preprocess_functions
from data_loaders import load_tensors, extract_frames_csl, EMGFrameLoader
from utils import majority_voting_segments, train_model, test_model
from networks import CapgMyoNet

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')

# Prepare data specific and hyperparameter settings for experiment
data = {'DIR': '../datasets/csl',
        'num_gestures': 26,
        'num_repetitions': 10,
        'input_shape': (7, 24),
        'extract_frames': extract_frames_csl,
        'fs': 2048,
        'subs': list(range(1, 6)),
        'sessions': list(range(1, 6))
        }

exp = {
    'num_epochs': 15,
    'lr': 0.001,
    'batch_size': 32,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'M': 32
}

if __name__ == '__main__':
    # Preinitialize metric arrays
    subs, sessions = [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    for sub in tqdm(data['subs']):
        for session in data['sessions']:
            # Load data for given subject/session
            print('\n SUBJECT #{} - SESSION #{}'.format(sub, session))
            sub_id = 'subject{}'.format(sub)
            session = 'session{}'.format(session)
            subs.append(sub)
            sessions.append(session)

            # Load EMG data
            X, Y = load_tensors(path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_gestures'],
                                 input_shape=data['input_shape'], fs=data['fs'], sessions=session)
            X_train = torch.flatten(X[:, :, :-1, :, :, :, :], end_dim=2) # get all but one repetition
            Y_train = torch.flatten(Y[:, :, :-1, :], end_dim=2) # get all but one repetition
            X_test = torch.flatten(X[:, :, -1, :, :, :, :], end_dim=2) # get one repetition
            Y_test = torch.flatten(Y[:, :, -1, :], end_dim=2) # get one repetition
            
            # Create torch dataloaders
            train_data = EMGFrameLoader(X=X_train, Y=Y_train)
            test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False)
            train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            # Model/training set-up
            model = CapgMyoNet(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_classes']).to(device)
            num_epochs = exp['num_epochs']
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=exp['num_epochs'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])

            # Train the model
            train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs']) # run training loop

            # Testing loop over test loader
            model.eval()
            with torch.no_grad():
                all_labs, all_preds = test_model(model, test_loader)

            acc = accuracy_score(all_labs, all_preds)
            accs.append(acc)
            print('Test Accuracy:', acc)

            # MAJORITY VOTING PREDICTIONS
            all_preds_maj = np.array(majority_voting_segments(all_preds, M=exp['32'], n_samples=train_data.n_samples))
            maj_acc = accuracy_score(all_labs, all_preds_maj)
            maj_accs.append(maj_acc)
            print('Majority Voting Test Accuracy:', maj_acc)

    # Save experiment data in .csv file
    data = np.array([subs, accs, sessions, maj_accs]).T
    df = pd.DataFrame(data=data, columns=['Subjects', 'Sessions', 'Accuracy', 'MV Accuracy'])
    df.to_csv('./baseline.csv')
