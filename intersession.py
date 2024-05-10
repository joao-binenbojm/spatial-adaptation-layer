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
from torchvision.transforms.v2 import RandomAffine, InterpolationMode, Compose
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS, CapgmyoDataSegmentRMS, CSLDataSegmentRMS
from torch_loaders import EMGFrameLoader
from utils.deep_learning import train_model, test_model
from networks import CapgMyoNet, CapgMyoNetInterpolate, RMSNet, median_pool_2d
from utils.emg_processing import majority_voting_full_segment, majority_voting_segments

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    exp = 'exp.json'

    # Experiment condition loading
    name = exp[:-5] # keep experiment name
    print('#'*40 + '\n\n' + 'EXPERIMENT:'+ name + '\n\n' + '#'*40)

    with open(exp) as f:
        exp = json.load(f)
    with open('{}.json'.format(exp['dataset'])) as f:
        data = json.load(f)
    emg_tensorizer_def = eval(exp['emg_tensorizer'])

    t0 = time()

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]
    subs, test_sessions, train_sessions, adapt_reps = [], [], [], []
    xshifts, yshifts = [], []
    accs, tuned_accs = [], [] # different metrics to be saved in csv from experiment
    maj_accs, maj_tuned_accs = [], [] 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('INTERSESSION:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        # Load data for given subject/session
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=session_ids, intrasession=False, Trms=exp['Trms'])
        emg_tensorizer.load_tensors()

        # Run code 5 times for every train/test session pair, except where same session is used for train and test
        for train_session in tqdm(data['sessions']):
            for test_session in tqdm(data['sessions']):
                # Only run if train/test session aren't the same
                if test_session == train_session:
                    continue
                
                for adapt_rep in range(10): # for each possible repetition we can use to adapt
                    subs.append(sub)
                    train_sessions.append(train_session)
                    test_sessions.append(test_session)
                    adapt_reps.append(adapt_rep)
                    print('\n SUBJECT #{}'.format(sub+1))
                    print('TEST SESSION #{}, TRAIN SESSION #{}'.format(test_session+1, train_session+1))
                    print('ADAPT REP #{}'.format(adapt_rep+1))
                    
                    X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(
                                                                                    test_session=test_session,
                                                                                    train_session=train_session,
                                                                                    rep_idx=adapt_rep)

                    # COMPUTE THE AVERAGE EMG IMAGE FOR EACH GESTURE
                    X_train_plot = median_pool_2d(X_train, kernel_size=(3,1), padding=(1,0))
                    # X_train_plot = X_train
                    if exp['dataset'] == 'csl':
                        fig, axs = plt.subplots(5, 5, figsize=(100, 100))
                        for idx in range(5):
                            for jdx in range(5):
                                cur_label = idx*5 + jdx
                                Xmean = X_train_plot[Y_train == cur_label, 0, :, :].mean(axis=0)
                                im_plot = axs[idx, jdx].imshow(Xmean, cmap='gray')
                                axs[idx, jdx].set_title(str(idx*5 + jdx))
                                plt.colorbar(im_plot, ax=axs[idx, jdx])
                        plt.savefig('avg_image.jpg')
                        plt.close()
                    elif exp['dataset'] == 'capgmyo':
                        fig, axs = plt.subplots(2, 4, figsize=(100, 100))
                        for idx in range(2):
                            for jdx in range(4):
                                cur_label = idx*4 + jdx
                                Xmean = X_train_plot[Y_train == cur_label, 0, :, :].mean(axis=0)
                                im_plot = axs[idx, jdx].imshow(Xmean, cmap='gray')
                                axs[idx, jdx].set_title(str(idx*4 + jdx))
                                plt.colorbar(im_plot, ax=axs[idx, jdx])
                        plt.savefig('avg_image.jpg')
                        plt.close()
                    
                    # Get PyTorch DataLoaders
                    train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
                    adapt_data = EMGFrameLoader(X=X_adapt, Y=Y_adapt, train=False, norm=exp['norm'], stats=train_data.stats)
                    test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                    train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                    adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
                    test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

                    # Model/training set-up
                    model = eval(exp['network'])(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_gestures']).to(device)
                    num_epochs = exp['num_epochs']
                    criterion = nn.CrossEntropyLoss(reduction='sum')
                    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    #                             lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

                    # Train the model
                    model.shift.xshift.requires_grad = False
                    model.shift.yshift.requires_grad = False
                    model.baseline.requires_grad = False
                    train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop
                    
                    # Testing loop over test loader (Zero-shot)
                    print('TESTING...')
                    model.eval()
                    with torch.no_grad():
                        all_labs, all_preds = test_model(model, test_loader)

                    acc = accuracy_score(all_labs, all_preds)
                    accs.append(acc)
                    print('Test Accuracy:', acc)

                    # Majority voting, with number of frames depending on dataset used
                    if exp['dataset'] == 'capgmyo':
                        maj_all_preds = majority_voting_segments(all_preds, Mmj=75, durations=test_durations)
                        maj_acc = accuracy_score(all_labs, maj_all_preds)
                        maj_accs.append(maj_acc)
                        print('Majority Voting Accuracy:', maj_acc)
                    
                    else: # if csl, compute one MJV predition for each test segment
                        maj_all_preds, maj_all_labs = (all_preds, test_durations), majority_voting_full_segment(all_labs, test_durations)
                        maj_acc = accuracy_score(maj_all_labs, maj_all_preds)
                        maj_accs.append(maj_acc)
                        print('Majority Voting Accuracy:', maj_acc)

                    # Fine-tune to update model's shifting position
                    print('FINE-TUNING...')
                    if exp['adaptation'] == 'shift-adaptation':
                        for param in model.parameters():
                            param.requires_grad = False
                        model.eval()
                        model.shift.xshift.requires_grad = True
                        model.shift.yshift.requires_grad = True
                        model.baseline.requires_grad = True
                
                    for g in optimizer.param_groups:
                        g['lr'] = exp['lr']
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(adapt_loader)*data['num_repetitions'])
                    train_model(model, adapt_loader, optimizer, criterion, num_epochs=exp['num_epochs']*data['num_repetitions'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop

                    xshift = model.shift.xshift.cpu().detach().numpy()[0]
                    yshift = model.shift.yshift.cpu().detach().numpy()[0]

                    xshifts.append(xshift)
                    yshifts.append(yshift)

                    # Testing loop over test loader (K-shot)
                    print('TESTING...')
                    model.eval()
                    with torch.no_grad():
                        print('LEARNED SHIFTS: x: {} , y: {}'.format(xshift, yshift))
                        all_labs, all_preds = test_model(model, test_loader)

                    tuned_acc = accuracy_score(all_labs, all_preds)
                    tuned_accs.append(tuned_acc)
                    print('Tuned Test Accuracy:', tuned_acc)

                    # Majority voting, with number of frames depending on dataset used
                    if exp['dataset'] == 'capgmyo':
                        maj_all_preds = majority_voting_segments(all_preds, Mmj=75, durations=test_durations)
                        maj_tuned_acc = accuracy_score(all_labs, maj_all_preds)
                        maj_tuned_accs.append(maj_tuned_acc)
                        print('Majority Voting Tuned Accuracy:', maj_tuned_acc)
                    
                    else: # if csl, compute one MJV predition for each test segment
                        maj_all_preds, maj_all_labs = majority_voting_full_segment(all_preds, test_durations), majority_voting_full_segment(all_labs, test_durations)
                        maj_tuned_acc = accuracy_score(maj_all_labs, maj_all_preds)
                        maj_tuned_accs.append(maj_tuned_acc)
                        print('Majority Voting Tuned Accuracy:', maj_tuned_acc)

                    # Get confusion matrix
                    labs = np.arange(data['num_gestures'])
                    cf = confusion_matrix(all_labs, all_preds, labels=labs)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                    disp.plot()
                    plt.savefig('cfm.jpg')
                    plt.close()

                    # SAVE RESULTS
                    arr = np.array([subs, train_sessions, test_sessions, adapt_reps, accs, tuned_accs, maj_accs, maj_tuned_accs, xshifts, yshifts]).T
                    df = pd.DataFrame(data=arr, columns=['Subjects', 'Train Sessions', 'Test Sessions', 'Adaptation Repetitions', 'Accuracy', 'Tuned Accuracy', 'Majority Voting Accuracy', 'Majority Voting Tuned Accuracy', 'xshift', 'yshift'])
                    df.to_csv(name + '.csv')

    # Save experiment data in .csv file
    arr = np.array([subs, train_sessions, test_sessions, adapt_reps, accs, tuned_accs, maj_accs, maj_tuned_accs, xshifts, yshifts]).T
    df = pd.DataFrame(data=arr, columns=['Subjects', 'Train Sessions', 'Test Sessions', 'Adaptation Repetitions', 'Accuracy', 'Tuned Accuracy', 'Majority Voting Accuracy', 'Majority Voting Tuned Accuracy','xshift', 'yshift'])
    df.to_csv(name + '.csv')

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
