import os
from time import time
import json
from copy import deepcopy
import sys
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torchvision.transforms.v2 import RandomAffine, InterpolationMode, Compose
from sklearn.metrics import  accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
import matplotlib.pyplot as plt
from tensorize_emg import CapgmyoData, CSLData, HyserData, GrabmyoData
from torch_loaders import EMGFrameLoader
from sal_classification.deep_learning import train_model, test_model, init_adabn, initial_search
from networks import CapgMyoNet, LogisticRegressor, VGG11Net, MobileNetV3SmallNet #, LogisticRegressorHyser
from networks_utils import median_pool_2d
from emg_processing import majority_voting_full_segment, majority_voting_segments

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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
learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
accs, tuned_accs = [], [] # different metrics to be saved in csv from experiment
mv_accs, mv_tuned_accs = [], []
f1_scores, tuned_f1_scores = [], []
mv_f1_scores, mv_tuned_f1_scores = [], []

is_model_trained = False
device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
print('Device:', device)

# Get labels
if exp['gest_subset']:
    nlabels = len(exp['gest_subset'])
else:
    nlabels = data['num_gestures']
cf_tot = np.zeros((nlabels, nlabels))

print('INTERSESSION:', data['dataset_name'])
print('CONDITIONS:', exp['name'])
# data['subs'] = [3,4]
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
            
            # if (train_session != 4) or (test_session != 0):
            #     continue

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
                                
                # Get adaptation gest from gest subset
                if exp['adapt_gest_subset'] is not None:
                    if exp['gest_subset']:
                        subgests = [exp['gest_subset'].index(gest) for gest in exp['adapt_gest_subset']] # get gesture indices from subset
                    else:
                        subgests = exp['adapt_gest_subset']
                else:
                    subgests = None
                X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors_intersession(
                                                                                test_session=test_idx,
                                                                                train_session=train_idx,
                                                                                rep_idx=int(adapt_rep),
                                                                                gest_idxs=subgests) # adapt to only one gesture
                
                
                # from scipy.ndimage import binary_dilation, label
                # from skimage.morphology import disk  # makes circular structuring element

                # def remove_small_clusters(mask, min_size=2):
                #     labeled, num = label(mask)
                #     sizes = np.bincount(labeled.ravel())
                #     keep = sizes >= min_size
                #     keep[0] = 0  # background
                #     return keep[labeled]

                # Get superset outlier mask and apply it to train, adapt and test sets
                # structure = np.ones((3,3)) # structure for morphological operations

                # # Apply morphological opening, then five iterations of square dilation to the outlier mask of adapt session
                # outlier_mask_train, outlier_mask_adapt = np.array(outlier_mask_train).squeeze(), np.array(outlier_mask_adapt).squeeze()
                
                # # Clean up outlier masks
                # outlier_mask_train = remove_small_clusters(outlier_mask_train)
                # outlier_mask_train = binary_dilation(outlier_mask_train, structure=structure, iterations=1)

                # outlier_mask_adapt = remove_small_clusters(outlier_mask_adapt)
                # structure = np.ones_like(disk(5))
                # outlier_mask_adapt_dilated = binary_dilation(outlier_mask_adapt, structure=structure, iterations=1)

                # fig, ax = plt.subplots(2,1)
                # ax[0].imshow(outlier_mask_train)
                # ax[0].set_title('Outliers (Train)')
                # ax[1].imshow(outlier_mask_adapt_dilated)
                # ax[1].set_title('Outliers (Adapt)')
                # plt.savefig('outlier-masks')
                # plt.close('all')

                # outlier_mask = np.logical_or(outlier_mask_train, outlier_mask_adapt_dilated) # combine train and adapt masks
                # # outlier_mask = np.logical_not(outlier_mask)
                # outlier_mask = torch.tensor(outlier_mask, dtype=torch.bool, device=X_train.device).unsqueeze(0).unsqueeze(0) # convert to tensor

                # # Apply outlier mask to train, adapt and test sets
                # # outlier_mask = outlier_mask.expand(X_train.shape)  # shape: (T, 1, H, W)

                ## HARD CODING OUTLIER MASK
                # outlier_mask = torch.ones((1,1,X_train.shape[2], X_train.shape[3]))
                # # outlier_mask[:,:,:2,:] = 0.0
                # outlier_mask[:,:,:2,:13] = 0.0
                # outlier_mask = outlier_mask.expand(X_train.shape).to(torch.bool)
                # valid_values = X_train[~outlier_mask]  # flattening valid values
                # mean = valid_values.mean()
                # std = valid_values.std()
                # noise = torch.randn_like(X_train) * std + mean
                # X_train[outlier_mask] = noise[outlier_mask]
                # X_train[X_train < 0] = 0.0
                
                # outlier_mask_adapt = binary_dilation(outlier_mask_adapt, structure=np.ones((3,3)), iterations=1)
                # outlier_mask_adapt = torch.tensor(outlier_mask_adapt, dtype=torch.bool, device=X_adapt.device).unsqueeze(0).unsqueeze(0) # convert to tensor
                # outlier_mask_adapt = outlier_mask_adapt.expand(X_adapt.shape)
                # valid_values = X_adapt[~outlier_mask_adapt]  # flattening valid values
                # mean = valid_values.mean()
                # std = valid_values.std()
                # noise_adapt = torch.randn_like(X_adapt) * std + mean
                # X_adapt[outlier_mask_adapt] = noise_adapt[outlier_mask_adapt]
                # X_adapt[X_adapt < 0] = 0.0

                # if exp['outlier_mask'] is not None:

                # # Handle outliers in the data
                # if exp['remove_outliers']:
                #     X_train = handle_outliers(X_train, Y_train)
                #     X_adapt = handle_outliers(X_adapt, Y_adapt)
                #     X_test = handle_outliers(X_test, Y_test)
                
                # Get test set image saved
                nlabels = max([len(Y_train.unique()), len(Y_test.unique())])
                nrows = max([math.ceil(nlabels / 4), 2])
                plt.figure()
                fig, ax = plt.subplots(nrows, 4)
                for idx in range(nrows):
                    for jdx in range(4):
                        label = idx*4 + jdx
                        ax[idx, jdx].imshow(X_train[Y_train==label,0,:,:].mean(dim=0))
                        ax[idx, jdx].axis('off')
                        ax[idx, jdx].set_title(f'Label: {label}')
                
                plt.savefig('baseline-outlier-removed.jpg')
                plt.close()

                # # Initialize transform for data augmentation
                # from transforms import RandomChannelCorruption, RandomChannelBlobCorruption
                # import torchvision
                # from torchvision.transforms import v2

                # # Oulier values
                # train_mean, train_std = X_train.mean(), X_train.std()
                # min_noise = train_mean + train_std
                # max_noise = train_mean + train_std*3

                # rms_transforms = torchvision.transforms.Compose([
                #     v2.GaussianNoise(sigma=0.1*train_std, mean=0.0, clip=False),
                #     v2.RandomAffine(
                #         degrees=15,
                #         translate=(5/X_train.shape[2], 5/X_train.shape[3]),
                #         scale=(0.9,1.1),
                #         shear=(-0.1,0.1)
                #     ),
                #     v2.GaussianBlur(kernel_size=3),
                #     v2.RandomErasing()
                #     # RandomChannelCorruption(n_channels=20, min_noise=min_noise, max_noise=max_noise)
                # ])

                # Get PyTortorch DataLoaders
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

                    H, W = X_train.shape[2], X_train.shape[3] 

                    # if 'grabmyo' in exp['dataset']:
                    #     data['input_shape'] = (1, data['input_shape'][1])
                    
                    # H, W = data['input_shape']
                    # if exp['dataset'] == 'hyser':
                    #     H = H // 2
                    
                    # Set-up SAL boundaries
                    boundaries = [[-2*5.0/(W-1), 2*5.0/(W-1)], [-2*5.0/(H-1), 2*5.0/(H-1)], [-15/180, 15/180],
                            [1/1.1, 1.1], [1/1.1, 1.1], [-0.1, 0.1], [-0.1, 0.1]]
                        
                    base_model = eval(exp['network'])(input_shape=(X_train.shape[2], X_train.shape[3]), 
                                                        num_classes=emg_tensorizer.num_gestures, nfeatures=X_train.shape[1], p_input=exp['p_input'], baseline=exp['learnable_baseline'], 
                                                        input_transform_name=input_transform_name, circular=exp["circular"], boundaries=boundaries).to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()),
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader)*exp['num_epochs']//5)
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

                # COMPUTE MAJORITY VOTING ON THE FLY
                with torch.no_grad():
                    t = 0
                    all_mode_labs, all_mode_preds = [], []
                    for d_idx in range(len(test_durations)):
                        dt = int(test_durations[d_idx])
                        if dt > 0: # if duration is 0, skip
                            cur_labs = all_labs[t:t+dt]
                            cur_preds = all_preds[t:t+dt]
                            mode_labs,_ = mode(cur_labs)
                            mode_preds,_ = mode(cur_preds)
                            all_mode_labs.append(mode_labs)
                            all_mode_preds.append(mode_preds)
                            t += dt
                    
                # Compute accuracy and F1-score for majority voting
                mv_acc = accuracy_score(all_mode_labs, all_mode_preds)
                mv_f1 = f1_score(all_mode_labs, all_mode_preds, average='macro')
                print('Majority Voting Test Accuracy:', mv_acc)
                print('Majority Voting Test F1-Score:', mv_f1)
                mv_accs.append(mv_acc)
                mv_f1_scores.append(mv_f1)
                    
                # Get test set image saved
                nlabels = max([len(Y_train.unique()), len(Y_test.unique())])
                nrows = max([math.ceil(nlabels / 4), 2])
                plt.figure()
                fig, ax = plt.subplots(nrows, 4)
                for idx in range(nrows):
                    for jdx in range(4):
                        label = idx*4 + jdx
                        ax[idx, jdx].imshow(X_adapt[Y_adapt==label,0,:,:].mean(dim=0))
                        ax[idx, jdx].axis('off')
                        ax[idx, jdx].set_title(f'Label: {label}')
                
                plt.savefig('baseline-session2.jpg')
                plt.close()

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

                elif exp['adaptation'] == 'scratch-training': # train from scratch
                    adapted_model = eval(exp['network'])(input_shape=(X_train.shape[2], X_train.shape[3]), 
                                                         num_classes=emg_tensorizer.num_gestures, p_input=exp['p_input'], baseline=exp['learnable_baseline'], 
                                                         input_transform_name=input_transform_name, circular=exp["circular"], boundaries=boundaries).to(device)
                    adapted_model.train()

                if exp['adaptation'] == 'adabatch':
                    init_adabn(adapted_model)

                if exp['learnable_baseline']:
                    adapted_model.baseline.requires_grad = True
                
                # TESTING ON THE FLY STATS ADAPTATION
                if exp['corrective_gain']:
                    mask = torch.isin(Y_train, Y_adapt.unique())
                    X_train_sub = X_train[mask].clone()
                    adapted_model.get_session_means(X_train_sub, X_adapt) # get stats for both X_train and X_adapt

                adapted_model.adaptation_phase = True # set model to adaptation phase

                if exp['adaptation'] == "spatial-adaptation":
                    labels = torch.unique(Y_adapt)
                    X_adapt_search = torch.zeros(len(labels), 1, X_adapt.shape[2], X_adapt.shape[3], device=device)
                    with torch.no_grad():
                        for idx, label in enumerate(labels):
                            X_adapt_search[idx, 0, :, :] = torch.sqrt((X_adapt[Y_adapt == label]**2).mean(dim=0))
                            # X_adapt_search = X_adapt[Y_adapt == label].mean(dim=0, keepdim=True)
                    adapt_search_data = EMGFrameLoader(X=X_adapt_search, Y=labels, train=False, norm=exp['norm'], stats=train_data.stats)
                    adapt_search_loader = DataLoader(adapt_search_data, batch_size=len(labels), shuffle=True)
                    adapted_model.input_transform.mode = 'bicubic'
                    print('INITIAL CONDITION SAMPLING...')
                    boundaries = torch.tensor([5.0, 5.0, 15/180, 0.1, 0.1, 0.1, 0.1]) # symmetric for each dimension about zero
                    
                    initial_search(adapted_model, adapt_search_loader, boundaries, exp['adaptation_params'], H=H, W=W, npoints=int(4**7)) #data['num_repetitions']*exp['num_epochs']//2) # find optimal initial condition
                    adapted_model.input_transform.mode = 'bilinear' # set mode to bilinear for training

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                                lr=0.01, weight_decay=exp['weight_decay'])
                    scheduler_params = exp['scheduler']['params']
                    scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 1.0, total_iters=len(adapt_loader)*data['num_repetitions']*exp['num_epochs']//5)
                    train_model(adapted_model, adapt_search_loader, optimizer, criterion, num_epochs=500, scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler, verbose=False) # run training loop

                elif exp['adaptation'] == 'adabatch':
                    with torch.no_grad():
                        train_model(adapted_model, adapt_loader, optimizer, criterion, num_epochs=1) # single forward pass per batch
                else:

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                                lr=exp['lr'], weight_decay=exp['weight_decay'])
                    scheduler_params = exp['scheduler']['params']
                    scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(adapt_loader)*data['num_repetitions']*exp['num_epochs']//5)
                    train_model(adapted_model, adapt_loader, optimizer, criterion, num_epochs=exp['num_epochs']*data['num_repetitions'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler, verbose=False) # run training loop

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
                adapted_model.eval()
                with torch.no_grad():
                    tuned_all_labs, tuned_all_preds = test_model(adapted_model, test_loader)

                tuned_acc = accuracy_score(tuned_all_labs, tuned_all_preds)
                tuned_f1 = f1_score(tuned_all_labs, tuned_all_preds, average='macro')
                tuned_accs.append(tuned_acc)    
                tuned_f1_scores.append(tuned_f1)
                print('Tuned Test Accuracy:', tuned_acc)
                print('Tuned Test F1-Score:', tuned_f1)

                # Compute majority voting on the fly
                with torch.no_grad():
                    t = 0
                    tuned_all_mode_labs, tuned_all_mode_preds = [], []
                    for d_idx in range(len(test_durations)):
                        dt = int(test_durations[d_idx])
                        if dt > 0: # if duration is 0, skip
                            cur_labs = tuned_all_labs[t:t+dt]
                            cur_preds = tuned_all_preds[t:t+dt]
                            mode_labs,_ = mode(cur_labs)
                            mode_preds,_ = mode(cur_preds)
                            tuned_all_mode_labs.append(mode_labs)
                            tuned_all_mode_preds.append(mode_preds)
                            t += dt 
                
                # Compute accuracy and F1-score for majority voting
                tuned_mv_acc = accuracy_score(tuned_all_mode_labs, tuned_all_mode_preds)
                tuned_mv_f1 = f1_score(tuned_all_mode_labs, tuned_all_mode_preds, average='macro')
                print('Tuned Majority Voting Test Accuracy:', tuned_mv_acc)
                print('Tuned Majority Voting Test F1-Score:', tuned_mv_f1)
                mv_tuned_accs.append(tuned_mv_acc)
                mv_tuned_f1_scores.append(tuned_mv_f1)

                # Get confusion matrix
                labs = np.arange(max([len(Y_train.unique()), len(Y_test.unique())]))
                cf = confusion_matrix(tuned_all_labs, tuned_all_preds, labels=labs)
                try:
                    cf_tot = cf_tot + cf
                except:
                    continue
                disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                disp.plot()
                plt.savefig('cfm.jpg')
                plt.close('all')

                # SAVE RESULTS
                data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Majority Voting Accuracy": mv_accs, "Tuned Accuracy": tuned_accs, "Majority Voting Tuned Accuracy": mv_tuned_accs,
                            'F1-Score': f1_scores, "Majority Voting F1-Score": mv_f1_scores, 'Tuned F1-Score': tuned_f1_scores, "Majority Voting Tuned F1-Score": mv_tuned_f1_scores}
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

                # if 'grabmyo' in exp['dataset']:
                #     data['input_shape'] = (2, data['input_shape'][1])
        is_model_trained = False

# Save experiment data in .csv file
data_dict = {"Subject": subs, "Train Sessions": train_sessions, "Test Sessions": test_sessions, "Adaptation Repetitions": adapt_reps,
                             "Accuracy": accs, "Majority Voting Accuracy": mv_accs, "Tuned Accuracy": tuned_accs, "Majority Voting Tuned Accuracy": mv_tuned_accs,
                            'F1-Score': f1_scores, "Majority Voting F1-Score": mv_f1_scores, 'Tuned F1-Score': tuned_f1_scores, "Majority Voting Tuned F1-Score": mv_tuned_f1_scores}
data_dict.update(learned_params)
df = pd.DataFrame(data_dict)
df.to_csv(f"{name}.csv")

disp = ConfusionMatrixDisplay(confusion_matrix=cf_tot, display_labels=labs)
disp.plot()
plt.savefig('cfm_tot.jpg')
plt.close()

# Initialize wandb and make sure no other runs are active concurrently for interference
while wandb.run is not None and not wandb.run._is_finished():
    time.sleep(3)

# Log wandb conditions
wandb.init(
    # set the wandb project where this run will be logged
    project=exp["project"],
    config=config,
    name=name,
    mode='disabled'
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
