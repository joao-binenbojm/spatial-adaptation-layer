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
from data_loaders import CSLFrameLoader
from utils_emg import majority_voting_segments
from networks import CapgMyoNet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/capgmyo')

def train_model(model, train_loader, optimizer, criterion, num_epochs=2, val_loader=None):
    '''Training loop for given experiment.'''
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.view(-1).type(torch.LongTensor).to(device)
            # forward pass
            outputs = model(signals).to(device)
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TENSORBOARD
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted.squeeze() == labels.view(-1)).sum().item()

            if (i + 1) % 100 == 0:
                print('Epoch {} / {}, step {} / {}, loss = {:4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.add_scalar('training loss', running_loss/100, epoch * len(train_loader) + i)
                writer.add_scalar('training accuracy', running_correct/100, epoch * len(train_loader) + i)
                running_loss = 0.0
                running_correct = 0

        # # After every epoch, compute what the test accuracy would have been
        # model.eval() # optional when not using Model Specific layer
        # all_preds, all_labs = [], []
        # if val_loader is not None: # if there is a validation set provided
        #     for signals, labels in test_loader:
        #         signals, labels = signals.to(device), labels.to(device).view(-1)
        #         target = model(signals)
        #         _,predictions = torch.max(target, 1) # get class labels
        #         all_preds.extend(predictions.tolist())
        #         all_labs.extend(labels.tolist())
        #     all_preds, all_labs = np.array(all_preds), np.array(all_labs)
        #     model.train() # setting model back to training mode
        #     print('###############################\n')
        #     print('Test Accuracy:', accuracy_score(all_labs, all_preds))
        #     print('Balanced Test Accuracy:', balanced_accuracy_score(all_labs, all_preds))
        #     print('\n###############################')

if __name__ == '__main__':
    
    # Predefine lists to take in different metrics
    DIR = '../datasets/csl'
    subs, sessions = [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    for sub in tqdm(range(1, 6)):
        for session in range(1, 6):
            # Load data for given subject/session
            print('\n SUBJECT #{} - SESSION #{}'.format(sub, session))
            sub_id = 'subject{}'.format(sub)
            session = 'session{}'.format(session)
            subs.append(sub)
            sessions.append(session)
            train_data = CSLFrameLoader(sub=sub_id, intrasession=True, session=session, test_rep=8)
            test_data = train_data.train_test_split()
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            # Model/training set-up
            model = CapgMyoNet(channels=168, input_shape=(7, 24), num_classes=26).to(device)
            num_epochs = 15
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

            # # TENSORBOARD
            # examples = iter(test_loader)
            # example_data, example_targets = next(examples)
            # writer.add_graph(model, example_data.to(device))
            # writer.close()
            # # sys.exit()
            # running_loss = 0.0
            # running_correct = 0

            train_model(model, train_loader, optimizer, criterion, num_epochs=num_epochs) # run training loop

            # TESTING
            all_labs, all_preds = [], []
            model.eval()
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for i, (signals, labels) in enumerate(test_loader):
                    signals, labels = signals.to(device), labels.view(-1).to(device)
                    outputs = model(signals).to(device)
                    _,predictions = torch.max(outputs, 1) # get class labels
                    all_labs.extend(labels.cpu().tolist())
                    all_preds.extend(predictions.cpu().tolist())

                acc = accuracy_score(all_labs, all_preds)
                accs.append(acc)
                print('Test Accuracy:', acc)

            # MAJORITY VOTING PREDICTIONS
            all_preds_maj = np.array(majority_voting_segments(all_preds, M=150, n_samples=train_data.n_samples))
            maj_acc = accuracy_score(all_labs, all_preds_maj)
            maj_accs.append(maj_acc)
            print('Majority Voting Test Accuracy:', maj_acc)

    # Save experiment data in .csv file
    data = np.array([subs, accs, sessions, maj_accs]).T
    df = pd.DataFrame(data=data, columns=['Subjects', 'Sessions', 'Accuracy', 'MV Accuracy'])
    df.to_csv('./bandstop.csv')
        