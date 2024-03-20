from collections import Counter
from scipy import signal
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/capgmyo')

## Majority Voting
def majority_voting(predictions, M=32):
    ''' Go over the length of the segment, and sequentially increase the count of a given symbol.
        This count is then removed when the window moves out of its location. This results in only
        tracking the counts of elements within the given window. Majority voting window is centered around
        present index, with M past and M future indices.
        Inputs:
            arr[ndarray]: input segment to be majority voted over
            M[int]: majority voting window size
        Returns:
            modes[list]: contains labels after majority voting windows
    '''
    counts = Counter()  # Count occurrences in the first W-1 elements
    modes = []  # List to store the modes for each window
    L = len(predictions)
    for i in range(-M, L): # goes up by 1 every time
        # Only begin reducing counts one index after a full voting window
        if i > M:
            counts[predictions[i - M - 1]] -= 1  # Decrease count of the element leaving the window
            if counts[predictions[i - M]] == 0:
                del counts[predictions[i - M]]  # Remove element from counts if count becomes 0
        # Increase count of (i+M)th element while there is still data left
        if i + M < L:
            counts[predictions[i + M]] += 1
        # Only begin adding modes from 0th index
        if i >= 0:
            mode = max(counts, key=counts.get)  # Calculate the mode for the current window
            modes.append(mode)

    return modes

def majority_voting_segments(predictions, M=32, n_samples=1000):
    ''' This function assumes that multiple repetition segments are concatenated together.
        Thus, it applies majority voting to each segment individually.
    '''
    voted_predictions = []
    for idx in range(len(predictions) // n_samples):
        preds = predictions[idx*n_samples : (idx+1)*n_samples] # extract segment
        voted_predictions_segment = majority_voting(preds, M=M) # get majority voting of given segment
        voted_predictions.extend(voted_predictions_segment) # add majority voted to final segment
    return voted_predictions

## FILTERING

def bandstop(data, fs=1000):
    '''Used to remove powerline interference and its multiples.'''
    sos = signal.butter(2, (45, 55), btype='bandstop', output='sos', fs=fs)
    data = signal.sosfilt(sos, data, axis=0)
    return data

def bandpass(data, fs=1000):
    '''Used to maintain only information in relevant anatomical range of sEMG activity.'''
    sos = signal.butter(2, (20, 380), btype='bandpass', output='sos', fs=fs)
    data = signal.sosfilt(sos, data, axis=0)
    return data

## TRAINING/TESTING
def train_model(model, train_loader, optimizer, criterion, num_epochs=2, val_loader=None):
    '''Training loop for given experiment.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
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

def test_model(model, test_loader):
    ''' Takes given PyTorch model and test DataLoader, and returns all labels and corresponding model predictions.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
    all_labs, all_preds = [], []
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = signals.to(device), labels.view(-1).to(device)
        outputs = model(signals).to(device)
        _,predictions = torch.max(outputs, 1) # get class labels
        all_labs.extend(labels.cpu().tolist())
        all_preds.extend(predictions.cpu().tolist())
    return all_labs, all_preds