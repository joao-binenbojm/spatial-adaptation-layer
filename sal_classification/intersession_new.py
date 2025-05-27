import os
from time import time
import json
from copy import deepcopy
import sys

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
from tensorize_emg import CapgmyoData, CSLData, HyserData, GrabmyoData #CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS
from torch_loaders import EMGFrameLoader
from sal_classification.deep_learning import train_model, test_model, init_adabn, initial_search
from networks import CapgMyoNet, LogisticRegressor #, LogisticRegressorHyser
from networks_utils import median_pool_2d
from emg_processing import majority_voting_full_segment, majority_voting_segments

# def handle_outliers(emg_grid):
#     '''Determine outlier channels, and replace them with average of neighbours.'''
#     # Determine coordinates of outliers
#     H, W = emg_grid.shape[2:]
#     emg_grid_var = emg_grid.mean(dim=[0,1])
#     Q1, Q3 = torch.quantile(emg_grid_var.flatten(), 0.25), torch.quantile(emg_grid_var.flatten(), 0.75)
#     IQR = Q3 - Q1
#     lower, upper = Q1 -1.5*IQR, Q3 + 1.5*IQR
#     y, x = torch.where(torch.logical_or(emg_grid_var >= upper, emg_grid_var <= lower)) # only keep non-noisy channel
#     y, x = y.tolist(), x.tolist()

#     idx = 0
#     while idx < len(y): # for each outlier
#         l,r,b,t = x[idx] != 0, x[idx] != W-1, y[idx] != H-1, y[idx] != 0
#         subgrid = emg_grid[:, :, y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten(start_dim=2, end_dim=3)
#         subgridvar = emg_grid_var[y[idx]-t:y[idx]+b+1, x[idx]-l:x[idx]+r+1].flatten()
#         subgrid = subgrid[:, :, torch.logical_and(subgridvar < upper, subgridvar > lower)] # remove outlier channels included
#         if subgrid.shape[2] < 1: # if less than 3 valid neighbours, try again after filling in more channels
#             y.append(y[idx])
#             x.append(x[idx])
#         else:
#             emg_grid[:,:,y[idx], x[idx]] = subgrid.mean(dim=2) # compute as average of neighbours
#         idx += 1

#     return emg_grid

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

# if __name__ == '__main__':

exp_name = sys.argv[1]  # First argument after script name
exp_config = f'./sal_classification/{exp_name}.json'

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

t0 = time()

# Preinitialize metric arrays
session_ids = ['session'+str(ses+1) for ses in data['sessions']]
subs, test_sessions, train_sessions, adapt_reps = [], [], [], []
# xshifts, yshifts, rot_thetas, xscales, yscales, xshears, yshears = [], [], [], [], [], [], []
learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
accs, tuned_accs = [], [] # different metrics to be saved in csv from experiment
f1_scores, tuned_f1_scores = [], []
is_model_trained = False
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
print('Device:', device)

print('INTERSESSION:', data['dataset_name'])
for idx, sub in tqdm(enumerate(data['subs'])):
    # Load data for given subject/session
    sub_id = 'subject{}'.format(sub+1)

    # Load EMG data in uniform format
    print('\nLOADING EMG TENSOR...')
    is_segment = exp['dataset'] == 'csl'
    emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                        input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, intrasession=False, Trms=exp['Trms'], 
                                        remove_baseline=exp['real_baseline'], median_filter=exp['median-filter'], gest_subset=exp['gest_subset'], is_segment=is_segment) # 7-15 for capgmyo, 0-9 for csl)
    emg_tensorizer.load_tensors()

    # Run code 5 times for every train/test session pair, except where same session is used for train and test
    for train_idx, train_session in tqdm(enumerate(data['sessions'])):
        for test_idx, test_session in tqdm(enumerate(data['sessions'])):
            # Only run if train/test session aren't the same
            if test_session == train_session:
                continue
            
            rep_idxs = list(range(data['num_repetitions'])) # get all repetition numbers
            sample_reps = list(np.random.choice(list(range(data['num_repetitions'])), replace=False, size=exp['K'])) # sample repetition numbers, ensuring we don't sample the same rep twice
            for rep in sample_reps: rep_idxs.remove(rep) # remove sampled repetitions from the list of all repetitions
            
            for adapt_rep in sample_reps: # for each possible repetition we can use to adapt
                subs.append(sub)
                train_sessions.append(train_session)
                test_sessions.append(test_session)
                adapt_reps.append(adapt_rep)
                print('\n SUBJECT #{}'.format(sub+1))
                print('TEST SESSION #{}, TRAIN SESSION #{}'.format(test_session+1, train_session+1))
                print('ADAPT REP #{}'.format(adapt_rep+1))
                                
                X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors_intersession(
                                                                                test_session=test_idx,
                                                                                train_session=train_idx,
                                                                                rep_idx=int(adapt_rep))
                
                # Get PyTorch DataLoaders
                train_data = EMGFrameLoader(X=X_train.clone(), Y=Y_train.clone(), norm=exp['norm'])
                adapt_data = EMGFrameLoader(X=X_adapt.clone(), Y=Y_adapt.clone(), train=False, norm=exp['norm'], stats=train_data.stats)
                test_data = EMGFrameLoader(X=X_test.clone(), Y=Y_test.clone(), train=False, norm=exp['norm'], stats=train_data.stats)
                train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
                test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

                # Model/training set-up (if it hasn't been trained before)
                num_epochs = exp['num_epochs']
                criterion = nn.CrossEntropyLoss()
                if not is_model_trained:
                    
                    # Set input transformation for adaptation in case of the Hyser dataset
                    input_transform_name = exp['adaptation']
                    if exp['adaptation'] == 'spatial-adaptation':
                        if exp['dataset'] == 'hyser': 
                            input_transform_name += '-hyser'
                        elif exp['dataset'] == 'grabmyo':
                            input_transform_name += '-grabmyo'

                    if 'grabmyo' in exp['dataset']:
                        data['input_shape'] = (1, data['input_shape'][1])
                    
                    H, W = data['input_shape']
                    if exp['dataset'] == 'hyser':
                        H = H // 2
                    
                    # Set-up SAL boundaries
                    if 'spatial-adaptation' in exp['adaptation']:
                        if 'grabmyo' in exp['dataset']:
                            boundaries = [[-2*2.5/(W-1), 2*2.5/(W-1)], [-1, 1], [-15/180, 15/180],
                                           [1/1.1, 1.1], [1/1.1, 1.1], [-0.1, 0.1], [-0.1, 0.1]]
                        else:
                            boundaries = [[-2*2.5/(W-1), 2*2.5/(W-1)], [-2*2.5/(H-1), 2*2.5/(H-1)], [-15/180, 15/180],
                                           [1/1.1, 1.1], [1/1.1, 1.1], [-0.1, 0.1], [-0.1, 0.1]]
                        
                    base_model = eval(exp['network'])(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=emg_tensorizer.num_gestures, 
                                                        p_input=exp['p_input'], baseline=exp['learnable_baseline'], input_transform_name=input_transform_name, 
                                                        circular=exp["circular"], boundaries=boundaries).to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()),
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

                    # Train the model
                    train_model(base_model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop
                    
                    is_model_trained = True
                
                # Testing loop over test loader (Zero-shot)
                print('TESTING...')
                base_model.eval()
                with torch.no_grad():
                    all_labs, all_preds = test_model(base_model, test_loader)
                    acc = accuracy_score(all_labs, all_preds)
                    f1 = f1_score(all_labs, all_preds, average='macro')
                
                accs.append(acc)
                f1_scores.append(f1)
                print('Test Accuracy:', acc)
                print('Test F1-Score:', f1)

                # # Get test set image saved
                # plt.figure()
                # fig, ax = plt.subplots(2, 6)
                # for idx in range(2):
                #     for jdx in range(6):
                #         label = idx*6 + jdx
                #         ax[idx, jdx].imshow(X_test[Y_test==label,0,:,:].mean(dim=0))
                #         ax[idx, jdx].axis('off')
                #         ax[idx, jdx].set_title(f'Label: {label}')
                
                # plt.savefig('baseline-session2.jpg')
                # plt.close()

                # Fine-tune to update model's shifting position
                adapted_model = deepcopy(base_model)
                print('FINE-TUNING...')
                if exp['adaptation'] == 'spatial-adaptation':
                    for param in adapted_model.parameters():
                        param.requires_grad = False
                    for param_name in exp['adaptation_params'].keys():
                        param = getattr(adapted_model.input_transform, param_name)
                        for p in param: p.requires_grad = exp['adaptation_params'][param_name]

                elif exp['adaptation'] == 'linear-layer':
                    for param in adapted_model.input_transform.parameters():
                        param.requires_grad = True
                    
                elif exp['adaptation'] == 'fine-tuning':
                    adapted_model.train()
                    for param in adapted_model.parameters(): # make all parameters trainable
                        param.requires_grad = True

                if exp['adabatch']:
                    init_adabn(adapted_model)

                if exp['learnable_baseline']:
                    adapted_model.baseline.requires_grad = True
                
                # TESTING ON THE FLY STATS ADAPTATION
                if exp['corrective_gain']:
                    adapted_model.get_session_means(X_train, X_adapt) # get stats for both X_train and X_adapt

                print('INITIAL CONDITION SAMPLING...')
                boundaries = torch.tensor([2.5, 2.5, 15/180, 0.1, 0.1, 0.1, 0.1]) # symmetric for each dimension about zero
                adapted_model.adaptation_phase = True # set model to adaptation phase
                with torch.no_grad():
                    scaling_factors = (X_train.median(dim=0, keepdim=True)[0] / (X_adapt.median(dim=0, keepdim=True)[0] + 1e-8))
                    print('Scaling factors:', scaling_factors.squeeze())
                    X_test_scaled = X_test * scaling_factors # normalize X_adapt to X_train mean
                # test_data_scaled = EMGFrameLoader(X=X_test_scaled, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                # test_loader_scaled = DataLoader(test_data_scaled, batch_size=exp['batch_size'], shuffle=False)
                
                initial_search(adapted_model, adapt_loader, boundaries, exp['adaptation_params'], H=H, W=W, npoints=data['num_repetitions']*exp['num_epochs']//2) # find optimal initial condition

                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                             lr=exp['lr'], weight_decay=exp['weight_decay'])
                scheduler_params = exp['scheduler']['params']
                scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
                scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(adapt_loader)*data['num_repetitions']*exp['num_epochs']//2)
                train_model(adapted_model, adapt_loader, optimizer, criterion, num_epochs=data['num_repetitions']*exp['num_epochs']//2, scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler, verbose=True) # run training loop

                # Fetch params
                if exp['adaptation'] == 'spatial-adaptation':
                    cur_learned_params = []
                    for nsal_idx in range(adapted_model.input_transform.nsals):
                        params = adapted_model.input_transform.get_constrained_params(nsal_idx)
                        params = [p.detach().cpu().clone() for p in params]
                        cur_learned_params.append(params)
                    cur_learned_params = torch.stack([torch.stack(row) for row in cur_learned_params]).T
                    
                    for param, param_name in zip(cur_learned_params, exp['adaptation_params'].keys()):
                        learned_params[param_name].append(param)
                else:
                    for key in learned_params.keys(): learned_params[key].append(0.0) # set fetched parameters to 0 if no spatial adaptation

                # Testing loop over test loader (K-shot)
                if exp['adaptation'] == 'spatial-adaptation':
                    adapted_model.input_transform.mode = 'bicubic'

                print('TESTING...')
                with torch.no_grad():
                    tuned_all_labs, tuned_all_preds = test_model(adapted_model, test_loader)

                tuned_acc = accuracy_score(tuned_all_labs, tuned_all_preds)
                tuned_f1 = f1_score(tuned_all_labs, tuned_all_preds, average='macro')
                tuned_accs.append(tuned_acc)    
                tuned_f1_scores.append(tuned_f1)
                print('Tuned Test Accuracy:', tuned_acc)
                print('Tuned Test F1-Score:', tuned_f1)

                # Get confusion matrix
                labs = np.arange(data['num_gestures'])
                cf = confusion_matrix(tuned_all_labs, tuned_all_preds, labels=labs)
                disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                disp.plot()
                plt.savefig('cfm.jpg')
                plt.close()
                # SAVE RESULTS
                data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Tuned Accuracy": tuned_accs, 'F1-Score': f1_scores, 'Tuned F1-Score': tuned_f1_scores}
                data_dict.update(learned_params)
                df = pd.DataFrame(data_dict)
                df.to_csv(f"{name}.csv")
                print(f'----------------------Affine learned params----------------------')
                for param_key in learned_params.keys():
                    print(f'The {param_key} is {learned_params[param_key][-1]}')
      
                # # Get final 'fixed' model images for comparison
                # with torch.no_grad():
                #     X_test_fixed = adapted_model.input_transform(X_test)
                # plt.figure()
                # fig, ax = plt.subplots(2, 6)
                # for idx in range(2):
                #     for jdx in range(6):
                #         label = idx*6 + jdx
                #         ax[idx, jdx].imshow(X_test_fixed[Y_test==label,0,:,:].mean(dim=0))
                #         ax[idx, jdx].axis('off')
                #         ax[idx, jdx].set_title(f'Label: {label}')
                
                # plt.savefig('baseline-fixed.jpg')
                # plt.close()

                if 'grabmyo' in exp['dataset']:
                    data['input_shape'] = (2, data['input_shape'][1])
        is_model_trained = False

# Save experiment data in .csv file
data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Tuned Accuracy": tuned_accs, 'F1-Score': f1_scores, 'Tuned F1-Score': tuned_f1_scores}
data_dict.update(learned_params)
df = pd.DataFrame(data_dict)
df.to_csv(f"{name}.csv")

# Initialize wandb and make sure no other runs are active concurrently for interference
while wandb.run is not None and not wandb.run._is_finished():
    time.sleep(3)

# Log wandb conditions
wandb.init(
    # set the wandb project where this run will be logged
    project=exp["project"],
    config=config,
    name=name,
    # mode='disabled',
)

table = wandb.Table(dataframe=df)
wandb.log({'complete_results': table})
# wandb.log({'performance_histogram': wandb.plot.histogram(table, "Majority Voting Tuned Accuracy",
#   title="Performance Distribution Across Dataset")})
wandb.log({'Accuracy': df['Accuracy'].mean()})
wandb.log({'Tuned Accuracy': df['Tuned Accuracy'].mean()})
# wandb.log({'Majority Voting Accuracy': df['Majority Voting Accuracy'].mean()})
# wandb.log({'Majority Voting Tuned Accuracy': df['Majority Voting Tuned Accuracy'].mean()})

# if exp['project'] == 'architecture-evaluation':
#     wandb.log({'inter_channels': base_model.inter_channels})
#     wandb.log({'conv_kernel_size': str(base_model.conv_kernel_size)})
#     wandb.log({'pool_kernel_size': str(base_model.pool_kernel_size)})

tf = time()
h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
wandb.log({'Time Ellapsed':f'{h}h, {m}min'})
wandb.finish()
