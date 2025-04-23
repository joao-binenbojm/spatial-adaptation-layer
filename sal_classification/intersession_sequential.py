import os
from time import time
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torchvision.transforms.v2 import RandomAffine, InterpolationMode, Compose
from sklearn.metrics import  accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
# from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS, CapgmyoDataSegmentRMS, CSLDataSegmentRMS
from tensorize_emg import CapgmyoData, CSLData, HyserData #CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS
from torch_loaders import EMGFrameLoader
from sal_classification.deep_learning import train_model, test_model, init_adabn
from networks import CapgMyoNet, LogisticRegressor, LogisticRegressorHyser
from networks_utils import median_pool_2d
from emg_processing import majority_voting_full_segment, majority_voting_segments


def handle_outliers(emg_grid):
    '''Determine outlier channels, and replace them with average of neighbours.'''
    # Determine coordinates of outliers
    H, W = emg_grid.shape[2:]
    emg_grid_var = emg_grid.mean(dim=[0,1])
    Q1, Q3 = torch.quantile(emg_grid_var.flatten(), 0.25), torch.quantile(emg_grid_var.flatten(), 0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 -1.5*IQR, Q3 + 1.5*IQR
    y, x = torch.where(torch.logical_or(emg_grid_var >= upper, emg_grid_var <= lower)) # only keep non-noisy channel
    y, x = y.tolist(), x.tolist()

    idx = 0
    while idx < len(y): # for each outlier
        l,r,b,t = x[idx] != 0, x[idx] != W-1, y[idx] != H-1, y[idx] != 0
        subgrid = emg_grid[:, :, y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten(start_dim=2, end_dim=3)
        subgridvar = emg_grid_var[y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten()
        subgrid = subgrid[:, :, torch.logical_and(subgridvar < upper, subgridvar > lower)] # remove outlier channels included
        if subgrid.shape[2] < 1: # if less than 3 valid neighbours, try again after filling in more channels
            y.append(y[idx])
            x.append(x[idx])
        else:
            emg_grid[:,:,y[idx], x[idx]] = subgrid.mean(dim=2) # compute as average of neighbours
        idx += 1

    return emg_grid


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

# if __name__ == '__main__':

exp_config = './sal_classification/exp.json'

# Experiment condition loading
print('#'*40 + '\n\n' + 'RUNNING INTERSESSION EXPERIMENT' + '\n\n' + '#'*40)

with open(exp_config) as f:
    exp = json.load(f)
with open('./sal_classification/{}.json'.format(exp['dataset'])) as f:
    data = json.load(f)
emg_tensorizer_def = eval(exp['emg_tensorizer'])
name = exp['name'] # keep experiment name

# Log wandb conditions
config = deepcopy(exp)
config['scheduler'] = json.dumps(config['scheduler'])
wandb.init(
    # set the wandb project where this run will be logged
    project=exp["project"],
    config=config,
    name=name,
    mode='disabled',
)

t0 = time()

# Preinitialize metric arrays
session_ids = ['session'+str(ses+1) for ses in data['sessions']]
subs, test_sessions, train_sessions, adapt_reps = [], [], [], []
# xshifts, yshifts, rot_thetas, xscales, yscales, xshears, yshears = [], [], [], [], [], [], []
learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
accs, tuned_accs = [], [] # different metrics to be saved in csv from experiment
# maj_accs, maj_tuned_accs = [], [] 
is_model_trained = False
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
print('Device:', device)

print('INTERSESSION:', data['dataset_name'])
for idx, sub in tqdm(enumerate(data['subs'])):
    # Load data for given subject/session
    sub_id = 'subject{}'.format(sub+1)

    # Load EMG data in uniform format
    print('\nLOADING EMG TENSOR...')
    emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=len(exp['gest_subset']), num_repetitions=data['num_repetitions'],
                                        input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, intrasession=False, Trms=exp['Trms'], 
                                        remove_baseline=exp['real_baseline'], gest_subset=exp['gest_subset']) # 7-15 for capgmyo, 0-9 for csl)
    emg_tensorizer.load_tensors()

    # Run code 5 times for every train/test session pair, except where same session is used for train and test
    for train_idx, train_session in tqdm(enumerate(data['sessions'])):
        for test_idx, test_session in tqdm(enumerate(data['sessions'])):
            # Only run if train/test session aren't the same
            if test_session == train_session:
                continue
            
            sample_reps = list(np.random.choice(list(range(data['num_repetitions'])), replace=False, size=exp['K'])) # sample repetition numbers, ensuring we don't sample the same rep twice
            for adapt_rep in sample_reps: # for each possible repetition we can use to adapt
                subs.append(sub)
                train_sessions.append(train_session)
                test_sessions.append(test_session)
                adapt_reps.append(adapt_rep)
                print('\n SUBJECT #{}'.format(sub+1))
                print('TEST SESSION #{}, TRAIN SESSION #{}'.format(test_session+1, train_session+1))
                print('ADAPT REP #{}'.format(adapt_rep+1))
                
                X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(
                                                                                test_session=test_idx,
                                                                                train_session=train_idx,
                                                                                rep_idx=adapt_rep)

                # Handle outliers
                X_train, X_adapt, X_test = handle_outliers(X_train), handle_outliers(X_adapt), handle_outliers(X_test)

                # Get PyTorch DataLoaders
                train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
                adapt_data = EMGFrameLoader(X=X_adapt, Y=Y_adapt, train=False, norm=exp['norm'], stats=train_data.stats)
                test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
                test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

                # Model/training set-up (if it hasn't been trained before)
                num_epochs = exp['num_epochs']
                criterion = nn.CrossEntropyLoss()
                if not is_model_trained:

                    base_model = eval(exp['network'])(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_gestures'], 
                                                        p_input=exp['p_input'], baseline=exp['learnable_baseline']).to(device)
                    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()),
                    #                        lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()),
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

                    # Train the model
                    if exp['dataset'] == 'hyser':
                        for param1, param2 in zip(base_model.spatial_adapt1.parameters(), base_model.spatial_adapt2.parameters()):
                            param1.requires_grad, param2.requires_grad = False, False
                    else:
                        for param in base_model.spatial_adapt.parameters():
                            param.requires_grad = False
                    base_model.baseline.requires_grad = False
                    base_model.scaling.requires_grad = False ## TESTING!! ## 
                    train_model(base_model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop
                    
                    is_model_trained = True

                    # Testing loop over test loader (Zero-shot)
                    print('TESTING...')
                    base_model.eval()
                    with torch.no_grad():
                        all_labs, all_preds = test_model(base_model, test_loader)
                        # all_labs, all_preds = test_model(base_model, train_loader)
                        acc = accuracy_score(all_labs, all_preds)
                
                accs.append(acc)
                print('Test Accuracy:', acc)

                    # # Majority voting, with number of frames depending on dataset used
                    # if exp['dataset'] == 'capgmyo':
                    #     maj_all_preds = majority_voting_segments(all_preds, Mmj=75, durations=test_durations)
                    #     maj_acc = accuracy_score(all_labs, maj_all_preds)
                    #     maj_accs.append(maj_acc)
                    #     print('Majority Voting Accuracy:', maj_acc)
                    
                    # else: # if csl, compute one MJV predition for each test segment
                    #     maj_all_preds, maj_all_labs = majority_voting_full_segment(all_preds, test_durations), majority_voting_full_segment(all_labs, test_durations)
                    #     maj_acc = accuracy_score(maj_all_labs, maj_all_preds)
                    #     maj_accs.append(maj_acc)
                    #     print('Majority Voting Accuracy:', maj_acc)
                    
                # else:
                #     print('Using previously trained model (same test accuracy as previously)...')
                #     accs.append(acc)
                #     maj_accs.append(maj_acc)
                #     print('Test Accuracy:', acc)
                #     print('Majority Voting Accuracy:', maj_acc)

                # Fine-tune to update model's shifting position
                adapted_model = deepcopy(base_model)
                # adapted_model.train()
                # adapted_model.input_dropout.eval()
                print('FINE-TUNING...')
                if exp['adaptation'] == 'spatial-adaptation':
                    for param in adapted_model.parameters():
                        param.requires_grad = False

                    for param_name in exp['adaptation_params'].keys():
                        if exp['dataset'] == 'hyser': # if hyser, we get two of each
                            param1 = getattr(adapted_model.spatial_adapt1, param_name)
                            param1.requires_grad = exp['adaptation_params'][param_name]
                            param2 = getattr(adapted_model.spatial_adapt2, param_name)
                            param2.requires_grad = exp['adaptation_params'][param_name]
                        else:
                            param = getattr(adapted_model.spatial_adapt, param_name)
                            param.requires_grad = exp['adaptation_params'][param_name]

                elif exp['adaptation'] == 'fine-tuning':
                    adapted_model.train()
                    for param in adapted_model.parameters(): # make all parameters trainable
                        param.requires_grad = True

                if exp['adabatch']:
                    init_adabn(adapted_model)

                # if exp['learnable_baseline']:
                #     adapted_model.baseline.requires_grad = True
                
                    # adapted_model.scaling.requires_grad = True
                
                # 2 independent optimization steps
                # adapted_model.train()
                print('SPATIAL ADAPTATION...')
                for idx in range(4):
                    if idx == 1:
                        print('BASELINE ADAPTATION...')
                        # After spatial adapt, adapt ONLY LBN
                        for param in adapted_model.parameters(): # freeze all parameters
                          param.requires_grad = False
                        if exp['learnable_baseline']:
                            adapted_model.baseline.requires_grad = True
                    elif idx == 2: # adapt scalings
                        print('SCALING ADAPTATION...')
                        if exp['learnable_baseline']:
                            adapted_model.baseline.requires_grad = False
                        adapted_model.scaling.requires_grad = True
                    elif idx == 3: # joint fine-tuning of both
                        print('JOINT ADAPTATION...')
                        # Final adaptation with all adaptable layers.
                        for param_name in exp['adaptation_params'].keys():
                            if exp['dataset'] == 'hyser': # if hyser, we get two of each
                                param1 = getattr(adapted_model.spatial_adapt1, param_name)
                                param1.requires_grad = exp['adaptation_params'][param_name]
                                param2 = getattr(adapted_model.spatial_adapt2, param_name)
                                param2.requires_grad = exp['adaptation_params'][param_name]
                            else:
                                param = getattr(adapted_model.spatial_adapt, param_name)
                                param.requires_grad = exp['adaptation_params'][param_name]

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler_params = exp['scheduler']['params']
                    scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(adapt_loader)*data['num_repetitions'])
                
                    # Adapt spatial layer 
                    train_model(adapted_model, adapt_loader, optimizer, criterion, num_epochs=exp['num_epochs']*data['num_repetitions'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler, verbose=False) # run training loop

                # Fetch params
                for param_name in exp['adaptation_params'].keys():
                    if exp['dataset'] == 'hyser':
                        param1 = getattr(adapted_model.spatial_adapt1, param_name).item()
                        param2 = getattr(adapted_model.spatial_adapt2, param_name).item()
                        param_vals = [param1, param2]
                        learned_params[param_name].append(param_vals)
                    else:
                        param = getattr(adapted_model.spatial_adapt, param_name)
                        learned_params[param_name].append(param.item())

                # Testing loop over test loader (K-shot)
                print('TESTING...')
                with torch.no_grad():
                    print('LEARNED SHIFTS: x: {} , y: {}'.format(learned_params['xshift'][-1], learned_params['yshift'][-1]))
                    tuned_all_labs, tuned_all_preds = test_model(adapted_model, test_loader)

                tuned_acc = accuracy_score(tuned_all_labs, tuned_all_preds)
                tuned_accs.append(tuned_acc)
                print('Tuned Test Accuracy:', tuned_acc)

                # # Majority voting, with number of frames depending on dataset used
                # if exp['dataset'] == 'capgmyo':
                #     maj_all_preds = majority_voting_segments(all_preds, Mmj=75, durations=test_durations)
                #     maj_tuned_acc = accuracy_score(all_labs, maj_all_preds)
                #     maj_tuned_accs.append(maj_tuned_acc)
                #     print('Majority Voting Tuned Accuracy:', maj_tuned_acc)
                
                # else: # if csl, compute one MJV predition for each test segment
                #     maj_all_preds, maj_all_labs = majority_voting_full_segment(all_preds, test_durations), majority_voting_full_segment(all_labs, test_durations)
                #     maj_tuned_acc = accuracy_score(maj_all_labs, maj_all_preds)
                #     maj_tuned_accs.append(maj_tuned_acc)
                #     print('Majority Voting Tuned Accuracy:', maj_tuned_acc)

                # Get confusion matrix
                labs = np.arange(data['num_gestures'])
                cf = confusion_matrix(tuned_all_labs, tuned_all_preds, labels=labs)
                disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                disp.plot()
                plt.savefig('cfm.jpg')
                plt.close()
                # SAVE RESULTS
                data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Tuned Accuracy": tuned_accs}
                data_dict.update(learned_params)
                # arr = np.array([subs, train_sessions, test_sessions, adapt_reps, accs, tuned_accs, xshifts, yshifts, rot_thetas, xscales, yscales, xshears, yshears]).T
                # df = pd.DataFrame(data=arr, columns=['Subjects', 'Train Sessions', 'Test Sessions', 'Adaptation Repetitions', 'Accuracy', 'Tuned Accuracy', 'xshift', 'yshift', 'rot_theta','xscale','yscale','xshear','yshear'])
                df = pd.DataFrame(data_dict)
                df.to_csv(f"{name}.csv")
                print(f'----------------------Affine learned params----------------------A')
                for param_key in learned_params.keys():
                    print(f'The {param_key} is {learned_params[param_key][-1]}')
                # print(f'The x shift is {adapted_model.spatial_adapt.xshift}')
                # print(f'The y shift is {adapted_model.spatial_adapt.yshift}')
                # print(f'The rotation theta angle is {adapted_model.spatial_adapt.rot_theta}')
                # print(f'The x scale is {adapted_model.spatial_adapt.xscale}')
                # print(f'The y scale is {adapted_model.spatial_adapt.yscale}')
                # print(f'The x shear is {adapted_model.spatial_adapt.xshear}')
                # print(f'The y shear is {adapted_model.spatial_adapt.yshear}')        
        is_model_trained = False

# Save experiment data in .csv file
data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Tuned Accuracy": tuned_accs}
data_dict.update(learned_params)
df = pd.DataFrame(data_dict)
df.to_csv(f"{name}.csv")

# # Log wandb conditions
# config = deepcopy(exp)
# config['scheduler'] = json.dumps(config['scheduler'])
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="intersession",
#     config=config,
#     mode='disabled',
# )

table = wandb.Table(dataframe=df)
wandb.log({'complete_results': table})
# wandb.log({'performance_histogram': wandb.plot.histogram(table, "Majority Voting Tuned Accuracy",
#   title="Performance Distribution Across Dataset")})
wandb.log({'Accuracy': df['Accuracy'].mean()})
wandb.log({'Tuned Accuracy': df['Tuned Accuracy'].mean()})
# wandb.log({'Majority Voting Accuracy': df['Majority Voting Accuracy'].mean()})
# wandb.log({'Majority Voting Tuned Accuracy': df['Majority Voting Tuned Accuracy'].mean()})

if exp['project'] == 'architecture-evaluation':
    wandb.log({'inter_channels': base_model.inter_channels})
    wandb.log({'conv_kernel_size': str(base_model.conv_kernel_size)})
    wandb.log({'pool_kernel_size': str(base_model.pool_kernel_size)})

tf = time()
h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
wandb.log({'Time Ellapsed':f'{h}h, {m}min'})
wandb.finish()
