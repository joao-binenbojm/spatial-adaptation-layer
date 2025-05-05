from torch import nn
import torch
from tqdm import tqdm
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


def add_noise_input_transform(model, std=0.01):
    with torch.no_grad():
        noise = torch.randn_like(model.input_transform.xshift) * std
        model.input_transform.xshift.add_(noise)
        model.input_transform.yshift.add_(noise)
        # for param in model.input_transform.parameters():
        #     noise = torch.randn_like(param) * std
        #     param.add_(noise)

def loss_sampling(data_loader, trained_model, T=[0.0, 0.0], bounds=[0.0, 0.0], batch_size=128, num_points=20):
    ''' Method used to sample the loss landscape.'''
    loss = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = trained_model.input_shape

    # Getting torch meshgrid
    if not T:
        Tx, Ty = 0.0, 0.0
    else:
        Tx, Ty = T
    

    bounds[0] = 2*bounds[0]/(W-1)
    bounds[1] = 2*bounds[1]/(H-1)
    
    xbounds, ybounds = bounds
    xs = torch.linspace(-torch.tensor(xbounds), torch.tensor(xbounds), num_points)
    xs,_ = xs.sort()
    ys = torch.linspace(-torch.tensor(ybounds), torch.tensor(ybounds), num_points)
    ys,_ = ys.sort()

    loss_arr = torch.zeros(num_points, num_points).to(device)
    
    # Sample parameters
    with torch.no_grad():
        for x_idx, x in tqdm(enumerate(xs)):
            for y_idx, y in enumerate(ys):
                trained_model.input_transform.xshift.copy_(x)
                trained_model.input_transform.yshift.copy_(y)

                # Process EMG data in batches for each grid point
                running_loss, it_count = 0.0, 0
                for i, (signals, labels) in enumerate(data_loader):
                    add_noise_input_transform(trained_model, std=0.0001) # Add noise
                    signals = signals.to(device)
                    labels = labels.view(-1).type(torch.LongTensor).to(device)
                    outputs = trained_model(signals).to(device)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    it_count += 1
                
                avg_loss = running_loss / it_count
                loss_arr[y_idx, x_idx] = avg_loss

    plt.figure()
    ax = sns.heatmap(np.array(loss_arr.cpu()))
    ax.set(xlabel='Circumferential Shifts (cm)', ylabel='Longitudinal Shifts (cm)')
    if T:
        ax.text(np.where(xs.cpu()>=-Tx)[0][0], 
                np.where(ys.cpu()>=-Ty)[0][0], 'X', 
            color='green', ha='center', va='center', fontsize=16)
    
    plt.savefig('loss_landscape.jpg')
    print()

    return loss_arr


import os
import sys
from time import time
import json
import gc
import wandb

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torchvision.transforms.v2 import RandomAffine, InterpolationMode, Compose
from sklearn.metrics import  accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import psutil
from copy import deepcopy

from tensorize_emg import CapgmyoData, CSLData, HyserData, GrabmyoData
from torch_loaders import EMGFrameLoader
from sal_classification.deep_learning import train_model, test_model
from networks import CapgMyoNet, LogisticRegressor
from sal_classification.simulation_utils import apply_affine, get_grid_distance



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    exp_name = sys.argv[1]  # First argument after script name
    exp_config = f'./sal_classification/{exp_name}.json'

    # Experiment condition loading
    print('#'*40 + '\n\n' + 'SIMULATED PERTURBATIONS LOSS LANDSCAPE' + '\n\n' + '#'*40)

    with open(exp_config) as f:
        exp = json.load(f)
    with open('./sal_classification/{}.json'.format(exp['dataset'])) as f:
        data = json.load(f)
    emg_tensorizer_def = eval(exp['emg_tensorizer'])
    name = exp['name']# keep experiment name

    # Log wandb conditions
    config = deepcopy(exp)
    config['scheduler'] = json.dumps(config['scheduler'])

    t0 = time()

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]
    learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('SIMULATED SPATIAL PERTURBATIONS:', data['dataset_name'])
    sub_id = 'subject1'
    train_idx = 0
    test_idx = 1
    adapt_rep = 9
    # Load EMG data in uniform format
    print('\nLOADING EMG TENSOR...')
    emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                        input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, intrasession=False, Trms=exp['Trms'], 
                                        remove_baseline=exp['real_baseline'], gest_subset=exp['gest_subset']) # 7-15 for capgmyo, 0-9 for csl)
    # emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
    #                                     input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, intrasession=True, Trms=exp['Trms'], 
    #                                     remove_baseline=exp['real_baseline'], gest_subset=exp['gest_subset']) # 7-15 for capgmyo, 0-9 for csl)
    emg_tensorizer.load_tensors()

    # X_train, Y_train, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(test_session=session, rep_idx=rep_idx)
    X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(
                                                                                test_session=test_idx,
                                                                                train_session=train_idx,
                                                                                rep_idx=adapt_rep)


    # Apply randomly sampled affine transformation to test set
    H, W = data['input_shape']
    # boundaries = [[-2*2.0/(W-1), 2*2.0/(W-1)], [-2*2.0/(H-1), 2*2.0/(H-1)], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    # samps = [torch.tensor(np.random.uniform(*bound)).to(torch.float32) for bound in boundaries] # sampled transformation parameters

    # with torch.no_grad(): X_test = apply_affine(X_test, *samps, mode='bicubic') # apply spatial transformation to test set to simulate a second electrode placement with identical ground truths

    # Get PyTorch DataLoaders
    train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
    # adapt_data = EMGFrameLoader(X=deepcopy(X_test), Y=deepcopy(Y_test), norm=exp['norm'], stats=train_data.stats)
    adapt_data = EMGFrameLoader(X=X_adapt, Y=Y_adapt, norm=exp['norm'], stats=train_data.stats)
    test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
    train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
    adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

    # Model/training set-up
    model = eval(exp['network'])(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_gestures'], p_input=exp['p_input']).to(device)
    num_epochs = exp['num_epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=exp['lr'], weight_decay=exp['weight_decay'])
    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

    # Initialize model for adaptation
    # Train the model
    train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                warmup_scheduler=warmup_scheduler) # run training loop

    # Compute distance before SAL correction
    # dist = get_grid_distance(X_test.shape, samps, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    # print('AVERAGE ELECTRODE DISTANCE BETWEEN GRIDS (CM):', f'{dist} cm')

    # Compute loss landscape
    model.eval()
    model.adaptation_phase = True
    loss_arr = loss_sampling(adapt_loader, model, T=None, bounds=[-3.0, 3.0], batch_size=exp['batch_size'], num_points=20)

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))

