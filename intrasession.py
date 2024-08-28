import os
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
from torchvision.transforms.v2 import RandomAffine, InterpolationMode, Compose
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import psutil
from copy import deepcopy


# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS, CSLDataSegmentRMS, CapgmyoDataSegmentRMS
from torch_loaders import EMGFrameLoader
from deep_learning import train_model, test_model
from emg_processing import majority_voting_segments, majority_voting_full_segment
from networks import CapgMyoNet, LogisticRegressor
from networks_utils import median_pool_2d

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    exp_config = 'exp.json'

    # Experiment condition loading
    print('#'*40 + '\n\n' + 'RUNNING INTRASESSION EXPERIMENT' + '\n\n' + '#'*40)

    with open(exp_config) as f:
        exp = json.load(f)
    with open('{}.json'.format(exp['dataset'])) as f:
        data = json.load(f)
    emg_tensorizer_def = eval(exp['emg_tensorizer'])
    name = exp['name']# keep experiment name

    # Log wandb conditions
    config = deepcopy(exp)
    config['scheduler'] = json.dumps(config['scheduler'])
    wandb.init(
        # set the wandb project where this run will be logged
        project="intrasession",
        config=config,
        # mode='disabled',
    )


    t0 = time()

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]
    subs, sessions, test_reps = [], [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('INTRASESSION:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        # Load data for given subject/session
        # dg = data['dgs'][idx]
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=session_ids, intrasession=True)
        emg_tensorizer.load_tensors()

        for session in tqdm(data['sessions']):
            sample_reps = list(np.random.choice(list(range(10)), replace=False, size=exp['K']))
            for test_idx in sample_reps:
                subs.append(sub)
                sessions.append(session)
                test_reps.append(test_idx)
                print('\n SUBJECT #{}, SESSION #{}'.format(sub + 1, session + 1))
                print('TEST REPETITION #{}'.format(test_idx + 1))

                X_train, Y_train, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(test_session=session, rep_idx=test_idx)

                # Get PyTorch DataLoaders
                train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
                test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

                # Model/training set-up
                Nv, Nh = emg_tensorizer.input_shape[0], emg_tensorizer.input_shape[1]
                model = eval(exp['network'])(channels=np.prod((Nv, Nh)), input_shape=(Nv, Nh), num_classes=data['num_gestures']).to(device)
                num_epochs = exp['num_epochs']
                criterion = nn.CrossEntropyLoss(reduction='sum')
                # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                #                             lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=exp['lr'], weight_decay=exp['weight_decay'])
                scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

                # Train the model
                if exp['adaptation'] == 'shift-adaptation':
                    model.shift.xshift.requires_grad = False
                    model.shift.yshift.requires_grad = False
                    if exp['learnable_baseline']:
                        model.baseline.requires_grad = False
                train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler) # run training loop
        
                # Testing loop over test loader
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
                    maj_all_preds, maj_all_labs = majority_voting_full_segment(all_preds, test_durations), majority_voting_full_segment(all_labs, test_durations)
                    maj_acc = accuracy_score(maj_all_labs, maj_all_preds)
                    maj_accs.append(maj_acc)
                    print('Majority Voting Accuracy:', maj_acc)

                # # Plotting confusion matrix to understand what's going on
                # # labs = np.arange(1, 27)
                # labs = np.arange(data['num_gestures'])
                # cf = confusion_matrix(all_labs, all_preds, labels=labs)
                # disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                # disp.plot()
                # plt.savefig('cfm.jpg')
                # plt.close()

                # # Plotting a prediction-label stream
                # plt.figure()
                # plt.plot(all_labs)
                # plt.plot(maj_all_preds)
                # plt.legend(['Labels', 'Predictions'])
                # plt.savefig('stream_maj.jpg')
                # plt.close()

                # plt.figure()
                # plt.plot(all_labs)
                # plt.plot(all_preds)
                # plt.legend(['Labels', 'Predictions'])
                # plt.savefig('stream.jpg')
                # plt.close()

                # SAVE RESULTS
                arr = np.array([subs, sessions, test_reps, accs, maj_accs]).T
                df = pd.DataFrame(data=arr, columns=['Subjects', 'Sessions', 'Test Repetitions', 'Accuracy', 'Majority Voting Accuracy'])
                df.to_csv(name)

    # Save experiment data in .csv file
    arr = np.array([subs, sessions, test_reps, accs, maj_accs]).T
    df = pd.DataFrame(data=arr, columns=['Subjects', 'Sessions', 'Test Repetitions', 'Accuracy', 'Majority Voting Accuracy'])
    df.to_csv(name)

    # Logging final results onto wandb 
    table = wandb.Table(dataframe=df)
    wandb.log({'complete_results': table})
    wandb.log({'Accuracy': df['Accuracy'].mean()})
    wandb.log({'Majority Voting Accuracy': df['Majority Voting Accuracy'].mean()})

    wandb.finish()

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
