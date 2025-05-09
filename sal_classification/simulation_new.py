import os
import sys
from time import time
import json
import sys
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
from sal_classification.deep_learning import train_model, test_model, initial_search
from networks import CapgMyoNet, LogisticRegressor
from sal_classification.simulation_utils import apply_affine, get_grid_distance


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    exp_name = sys.argv[1]  # First argument after script name
    exp_config = f'./sal_classification/{exp_name}.json'

    # Experiment condition loading
    print('#'*40 + '\n\n' + 'RUNNING SIMULATED SPATIAL PERTURBATIONS' + '\n\n' + '#'*40)

    with open(exp_config) as f:
        exp = json.load(f)
    with open('./sal_classification/{}.json'.format(exp['dataset'])) as f:
        data = json.load(f)
    emg_tensorizer_def = eval(exp['emg_tensorizer'])
    name = exp['name']# keep experiment name

    # Log wandb conditions
    config = deepcopy(exp)
    config['scheduler'] = json.dumps(config['scheduler'])
    wandb.init(
        # set the wandb project where this run will be logged
        project=exp.pop("project"),
        config=config,
        name=name,
        mode='disabled',
    )

    t0 = time()

    # # Fake EMG grid
    # emg_grid = torch.zeros((1, 1, 7, 24))
    # emg_grid[0, 0, 3:5, 8:16] = 1.0
    # emg_transform = apply_affine(emg_grid, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0) # apply identity transformation to get original grid
    # d = get_grid_distance((1, 1, 7, 24), [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    # # d = get_grid_distance((1, 1, 7, 24), samps, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    npoints = 0

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]
    subs, sessions, test_reps = [], [], []
    learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
    true_params = {f'{key}_true' : [] for key in learned_params.keys()} 
    dists, corrected_dists = [], []
    accs, trans_accs, oracle_accs, tuned_accs = [], [], [], [] # different metrics to be saved in csv from experiment
    f1_scores, trans_f1_scores, oracle_f1_scores, tuned_f1_scores = [], [], [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('SIMULATED SPATIAL PERTURBATIONS:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        print('\nLOADING EMG TENSOR...')
        emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                        input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, Trms=exp['Trms'], 
                                        remove_baseline=exp['real_baseline'], gest_subset=exp['gest_subset'], intrasession=True) # 7-15 for capgmyo, 0-9 for csl)
        emg_tensorizer.load_tensors()


        for session in tqdm(data['sessions']):
            rep_idxs = list(range(data['num_repetitions'])) # get all repetition numbers
            sample_reps = list(np.random.choice(list(range(data['num_repetitions'])), replace=False, size=1))[0] # sample repetition numbers, ensuring we don't sample the same rep twice
            subs.append(sub)
            sessions.append(session)
            test_reps.append(sample_reps)
            print('\n SUBJECT #{}, SESSION #{}'.format(sub + 1, session + 1))
            print('TEST REPETITION #{}'.format(sample_reps))

            # X_train, Y_train, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(test_session=session, rep_idx=test_idx, )
            # val_idx = np.random.choice((rep_idxs), replace=False, size=1)[0] # sample a repetition for validation
            X_train, Y_train, X_test, Y_test, test_durations = emg_tensorizer.get_tensors(
                                                                            test_session=session,
                                                                            rep_idx=sample_reps,
                                                                            flatten=False)
            
            X_adapt = torch.flatten(X_train[:,[-1],:, :, :, :], end_dim=-4)
            X_test = torch.flatten(X_test, end_dim=-4)
            X_train = torch.flatten(X_train[:, :-1, :, :, :, :], end_dim=-4)
            Y_adapt = torch.flatten(Y_train[:, [-1], :], end_dim=-1)
            Y_train = torch.flatten(Y_train[:, :-1, :], end_dim=-1)
            Y_test = torch.flatten(Y_test, end_dim=-1)

            ## TODO: ADD OPTION IN TENSORIZER TO OBTAIN TENSORS UNFLATTENED!
            # Extract single final repetition of each gesture from training set to be the transformed adaptation set
            # indices = []
            # for i in range(emg_tensorizer.num_gestures):
            #     start = ((i * (emg_tensorizer.num_repetitions - 1)) + (emg_tensorizer.num_repetitions - 2)) * emg_tensorizer.num_samples
            #     end = start + emg_tensorizer.num_samples
            #     indices.append(torch.arange(start, end))
            # indices = torch.cat(indices)

            # X_adapt, Y_adapt = X_train[indices], Y_train[indices]
            # mask = torch.ones(X_train.shape[0], dtype=torch.bool)
            # mask[indices] = False
            # X_train, Y_train = X_train[mask], Y_train[mask] # remove from training set the repetition used for adaptatin

            # Pytorch training set and non-transformed test set
            train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
            test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
            train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

            # Train original classifier
            model = eval(exp['network'])(channels=np.prod(data['input_shape']), input_shape=data['input_shape'], num_classes=data['num_gestures'], p_input=exp['p_input']).to(device)
            num_epochs = exp['num_epochs']
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=exp['lr'], weight_decay=exp['weight_decay'])
            scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

            train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                        warmup_scheduler=warmup_scheduler) # run training loop

            # Record performance drop after spatial perturbation (zero-shot)
            print('TESTING PRIOR TRANSFORMATION...')
            model.eval()
            with torch.no_grad():
                all_labs, all_preds = test_model(model, test_loader)

            acc = accuracy_score(all_labs, all_preds)
            f1 = f1_score(all_labs, all_preds, average='macro')
            accs.append(acc)
            f1_scores.append(f1)
            print('Test Accuracy:', acc)
            print('Test F1 Score:', f1)

            # Apply randomly sampled affine transformation to test set
            H, W = data['input_shape']
            boundaries = [[-2*2.5/(W-1), 2*2.5/(W-1)], [-2*2.5/(H-1), 2*2.5/(H-1)], [-15/180, 15/180], [0.9, 1.1], [0.9, 1.1], [-0.1, 0.1], [-0.1, 0.1]]
            samps = [torch.tensor(np.random.uniform(*bound)).to(torch.float32) for bound in boundaries] # sampled transformation parameters
            model.true_params = samps

            with torch.no_grad(): 
                X_test = apply_affine(X_test, *samps, mode='bicubic') # apply spatial transformation to test set to simulate a second electrode placement with identical ground truths
                X_adapt = apply_affine(X_adapt, *samps, mode='bicubic') # apply spatial transformation to test set to simulate a second electrode placement with identical ground truths
            for idx, key in enumerate(true_params.keys()): true_params[key].append(samps[idx].item()) # track ground truth params

            # Apply transforms and reload data loaders
            test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
            adapt_data = EMGFrameLoader(X=X_adapt, Y=Y_adapt, train=False, norm=exp['norm'], stats=train_data.stats)
            adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

            # Record performance prior to adaptation
            print('TESTING POST TRANSFORMATION...')
            model.eval()
            with torch.no_grad():
                all_labs, all_preds = test_model(model, test_loader)

            trans_acc = accuracy_score(all_labs, all_preds)
            trans_f1 = f1_score(all_labs, all_preds, average='macro')
            trans_accs.append(trans_acc)
            trans_f1_scores.append(trans_f1)
            print('Transformed Test Accuracy:', trans_acc)
            print('Transformed F1 Score:', trans_f1)

            # Compute distance before SAL correction
            dists.append(get_grid_distance(X_test.shape, samps, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
            print('AVERAGE ELECTRODE DISTANCE BETWEEN GRIDS (CM):', f'{dists[-1]} cm')

            # Obtain oracle performance
            adapted_model = deepcopy(model)
            adapted_model.adaptation_phase = True
            adapted_model.input_transform.xshift.data = -samps[0]
            adapted_model.input_transform.yshift.data = -samps[1]
            adapted_model.input_transform.rot_theta.data = -samps[2]
            adapted_model.input_transform.xscale.data = 1.0/samps[3]
            adapted_model.input_transform.yscale.data = 1.0/samps[4]
            adapted_model.input_transform.xshear.data = -samps[5]
            adapted_model.input_transform.yshear.data = -samps[6]

            oracle_dist = get_grid_distance(X_test.shape, samps, [-samps[0], -samps[1], -samps[2], 1.0/samps[3], 1.0/samps[4], -samps[5], -samps[6]])

            # Record performance prior to adaptation
            print('TESTING WITH ORACLE OPTIMAL PARAMETERS...')
            adapted_model.eval()
            with torch.no_grad():
                all_labs, all_preds = test_model(adapted_model, test_loader)

            oracle_acc = accuracy_score(all_labs, all_preds)
            oracle_f1 = f1_score(all_labs, all_preds, average='macro')
            oracle_accs.append(oracle_acc)
            oracle_f1_scores.append(oracle_f1)
            print('Oracle Test Accuracy:', oracle_acc)
            print('Oracle F1 Score:', oracle_f1)

            # Reset SAL parameters
            adapted_model.input_transform.xshift.data = torch.tensor(0.0)
            adapted_model.input_transform.yshift.data = torch.tensor(0.0)
            adapted_model.input_transform.rot_theta.data = torch.tensor(0.0)
            adapted_model.input_transform.xscale.data = torch.tensor(1.0)
            adapted_model.input_transform.yscale.data = torch.tensor(1.0)
            adapted_model.input_transform.xshear.data = torch.tensor(0.0)
            adapted_model.input_transform.yshear.data = torch.tensor(0.0)

            # Spatially adapt the model's transformed position
            print('FINE-TUNING...')
            adapted_model = deepcopy(model)
            adapted_model.adaptation_phase = True
            for param in adapted_model.parameters():
                param.requires_grad = False
                for param_name in exp['adaptation_params'].keys():
                    param = getattr(adapted_model.input_transform, param_name)
                    if isinstance(param, nn.ParameterList):
                        for p in param: p.requires_grad = exp['adaptation_params'][param_name]
                    else:
                        param.requires_grad = exp['adaptation_params'][param_name]

            print('INITIAL CONDITION SAMPLING...')
            boundaries = torch.tensor([2.5, 2.5, 15/180, 0.1, 0.1, 0.1, 0.1]) # symmetric for each dimension about zero
            initial_search(adapted_model, adapt_loader, boundaries, npoints=npoints) # find optimal initial condition

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                lr=exp['lr'], weight_decay=exp['weight_decay'])
            scheduler_params = exp['scheduler']['params']
            scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
            scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 1.0, total_iters=len(test_loader)*data['num_repetitions'])

            # Adapt to given test set
            train_model(adapted_model, adapt_loader, optimizer, criterion, num_epochs=exp['num_epochs']*data['num_repetitions'], scheduler=scheduler,
                        warmup_scheduler=warmup_scheduler, simulation=True) # run training loop

            # Store learned params for later evaluation
            cur_learned_params = []
            for param_name in exp['adaptation_params'].keys():
                param = getattr(adapted_model.input_transform, param_name)
                if isinstance(param, nn.ParameterList): learned_params[param_name].append([p.item() for p in param])
                else: 
                    learned_params[param_name].append(param.item())
                    cur_learned_params.append(param.item())
            
            # Testing loop over test loader (K-shot)
            print('TESTING...')
            with torch.no_grad():
                all_labs, all_preds = test_model(adapted_model, test_loader)

            tuned_acc = accuracy_score(all_labs, all_preds)
            tuned_f1 = f1_score(all_labs, all_preds, average='macro')
            tuned_accs.append(tuned_acc)
            tuned_f1_scores.append(tuned_f1)
            print('Tuned Test Accuracy:', tuned_acc)
            print('Tuned Test F1 Score:', tuned_f1)

            print(f'----------------------Affine learned params----------------------')
            for param_key in learned_params.keys():
                print(f'The {param_key} is {learned_params[param_key][-1]}')

            # Compute average electrode distance between learned and true grid
            corrected_dists.append(get_grid_distance(X_test.shape, samps, cur_learned_params))
            print('AVERAGE ELECTRODE DISTANCE BETWEEN GRIDS (CM):', f'{corrected_dists[-1]} cm')

            # SAVE RESULT
            data_dict = {'Subjects': subs, 'Sessions':sessions, 'Test Repetitions':test_reps, 'Accuracy':accs, 'Transformed Accuracy': trans_accs, 'Oracle Accuracy': oracle_accs, 'Tuned Accuracy':tuned_accs, 
                         'F1-Score': f1_scores, 'Transformed F1-Score': trans_f1_scores, 'Oracle F1-Score': oracle_f1_scores, 'Tuned F1-Score': tuned_f1_scores, 'Distance (cm)':dists, 'Tuned Distance (cm)':corrected_dists}
            data_dict.update(true_params)
            data_dict.update(learned_params)
            df = pd.DataFrame(data_dict)
            df.to_csv(f"{name}.csv")

    # Save final experiment data in .csv file
            data_dict = {'Subjects': subs, 'Sessions':sessions, 'Test Repetitions':test_reps, 'Accuracy':accs, 'Transformed Accuracy': trans_accs, 'Oracle Accuracy': oracle_accs, 'Tuned Accuracy':tuned_accs, 
                         'F1-Score': f1_scores, 'Transformed F1-Score': trans_f1_scores, 'Oracle F1-Score': oracle_f1_scores, 'Tuned F1-Score': tuned_f1_scores, 'Distance (cm)':dists, 'Tuned Distance (cm)':corrected_dists}
    data_dict.update(true_params)
    data_dict.update(learned_params)
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{name}.csv")

    # # Logging final results onto wandb 
    # table = wandb.Table(dataframe=df)
    # wandb.log({'complete_results': table})
    # wandb.log({'Tuned Accuracy': df['Tuned Accuracy'].mean()})
    # wandb.log({'Majority Voting Tuned Accuracy': df['Majority Voting Tuned Accuracy'].mean()})


    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
    wandb.log({'Time Ellapsed':f'{h}h, {m}min'})
    wandb.finish()

