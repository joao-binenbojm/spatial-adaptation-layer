import os
import sys

import scipy
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score

import preprocess_functions as preprocess_functions
from new_data_loader import CapgMyoSubjectLoader, intrasession_load, majority_voting
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
                print('Epoch {} / {}, step {} / {}, loss = {:4f}'.format(epoch+1, num_epochs, i+1, len(train), loss.item()))
                writer.add_scalar('training loss', running_loss/100, epoch * len(train_loader) + i)
                writer.add_scalar('accuracy', running_correct/100, epoch * len(train_loader) + i)
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
    DIR = '../datasets/capgmyo/dba'
    subs = []
    accs, bal_accs, maj_accs, maj_bal_accs = [], [], [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    for sub in tqdm(range(1, 19)):
        # Load data for given subject
        print('\n SUBJECT #00{}'.format(sub))
        subs.append(sub)
        X_train, y_train, X_test, y_test = intrasession_load(DIR=DIR, sub=sub, wlen=1, stride=1)
        train = CapgMyoSubjectLoader(X=X_train, Y=y_train, window=1, stride=1)
        test = CapgMyoSubjectLoader(X=X_test, Y=y_test, window=1, stride=1)
        train_loader = DataLoader(train, batch_size=64, shuffle=True)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)

        # Get class weights based on class imbalance
        class_weights = torch.Tensor(train.weights).to(device) # to account for imbalanced data

        # Model/training set-up
        model = CapgMyoNet(channels=128, num_classes=9).to(device)
        num_epochs = 2
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

        # # TENSORBOARD
        # examples = iter(test_loader)
        # example_data, example_targets = next(examples)
        # writer.add_graph(model, example_data.to(device))
        # writer.close()
        # # sys.exit()
        # running_loss = 0.0
        # running_correct = 0

        train_model(model, train_loader, optimizer, criterion, val_loader=test_loader) # run training loop

        # TESTING
        all_labs, all_preds = [], []
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for i, (signals, labels) in enumerate(test_loader):
                signals = signals.to(device)
                labels = labels.view(-1).to(device)
                outputs = model(signals).to(device)
                _,predictions = torch.max(outputs, 1) # get class labels

                # Store predictions and labels
                all_labs.extend(labels.tolist())
                all_preds.extend(predictions.tolist())

        acc = accuracy_score(all_labs, all_preds)
        accs.append(acc)
        bal_acc = balanced_accuracy_score(all_labs, all_preds)
        bal_accs.append(bal_acc)
        print('Test Accuracy:', acc)
        print('Balanced Test Accuracy:', bal_acc)

        # MAJORITY VOTING PREDICTIONS
        all_preds_maj = np.array(majority_voting(all_preds, M=128))
        maj_acc = accuracy_score(all_labs, all_preds_maj)
        maj_accs.append(maj_acc)
        maj_bal_acc = balanced_accuracy_score(all_labs, all_preds_maj)
        maj_bal_accs.append(maj_bal_acc)
        print('Majority Voting Test Accuracy:', maj_acc)
        print('Majority Voting Balanced Test Accuracy:', maj_bal_acc)

    # Save experiment data in .csv file
    data = np.array([subs, accs, bal_accs, maj_accs, maj_bal_accs]).T
    df = pd.DataFrame(data=data, columns=['Subjects', 'Accuracy', 'Balanced Accuracy', 'MV Accuracy', 'MV Balanced Accuracy'])
    df.to_csv('./run.csv')
        