import os
import sys
from time import time
import json
import sys
import wandb
import numpy as np
import pandas as pd
from scipy.stats import mode
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

from tensorize_emg import CSLData, CapgmyoData
from torch_loaders import EMGFrameLoader
from sal_classification.deep_learning import train_model, test_model, initial_search
from networks import CapgMyoNet, LogisticRegressor
from networks_utils import median_pool_2d
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

    t0 = time() # start tracking time

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses+1) for ses in data['sessions']]
    subs, sessions, test_reps = [], [], []
    learned_params = {key: [] for key in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale', 'xshear', 'yshear']}
    true_params = {f'{key}_true' : [] for key in learned_params.keys()} 
    dists, corrected_dists = [], []
    accs, trans_accs, oracle_accs, tuned_accs = [], [], [], [] # different metrics to be saved in csv from experiment
    f1_scores, trans_f1_scores, oracle_f1_scores, tuned_f1_scores = [], [], [], []
    # mv_accs, mv_tuned_accs = [], []
    # f1_scores, tuned_f1_scores = [], []
    # mv_f1_scores, mv_tuned_f1_scores = [], []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 


    print('SIMULATED SPATIAL PERTURBATIONS:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        print('\nLOADING EMG TENSOR...')
        is_segment = exp['dataset'] == 'csl'
        emg_tensorizer = emg_tensorizer_def(dataset=exp['dataset'], path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                        input_shape=data['input_shape'], fs=data['fs'], rep_duration=data['rep_duration'], sessions=session_ids, Trms=exp['Trms'], 
                                        remove_baseline=exp['real_baseline'], median_filter=exp['median-filter'], gest_subset=exp['gest_subset'], is_segment=is_segment) # 7-15 for capgmyo, 0-9 for csl)
        
        emg_tensorizer.load_tensors()

        for session in tqdm(data['sessions']):
            rep_idxs = list(range(data['num_repetitions'])) # get all repetition numbers
            sample_reps = list(np.random.choice(list(range(data['num_repetitions'])), replace=False, size=2)) # sample repetition numbers, ensuring we don't sample the same rep twice
            subs.append(sub)
            sessions.append(session)
            test_reps.append(sample_reps)
            print('\n SUBJECT #{}, SESSION #{}'.format(sub + 1, session + 1))
            print('TEST REPETITION #{}'.format(sample_reps))

            X_train, Y_train, X_adapt, Y_adapt, X_test, Y_test, test_durations = emg_tensorizer.get_tensors_simulation(
                                                                            session=session,
                                                                            adapt_rep_idx=sample_reps[0],
                                                                            test_rep_idx=sample_reps[1])
            
            # Pytorch training set and non-transformed test set
            train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
            test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
            train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=False)

            # Train original classifier
            input_transform_name = exp['adaptation']
            input_transform_name = exp['adaptation']
            if exp['adaptation'] == 'spatial-adaptation':
                if exp['dataset'] == 'hyser': 
                    input_transform_name += '-hyser'
                elif exp['dataset'] == 'grabmyo':
                    input_transform_name += '-grabmyo'

            H, W = X_train.shape[2], X_train.shape[3] 
            
            # Set-up SAL boundaries
            boundaries = [[-2*2.5/(W-1), 2*2.5/(W-1)], [-2*2.5/(H-1), 2*2.5/(H-1)], [-15/180, 15/180],
                        [1/1.1, 1.1], [1/1.1, 1.1], [-0.1, 0.1], [-0.1, 0.1]]

            model = eval(exp['network'])(input_shape=(X_train.shape[2], X_train.shape[3]), 
                                         num_classes=emg_tensorizer.num_gestures, p_input=exp['p_input'], 
                                         baseline=exp['learnable_baseline'], input_transform_name=input_transform_name,
                                         circular=exp['circular'], boundaries=boundaries).to(device)
            num_epochs = exp['num_epochs']
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=exp['lr'], weight_decay=exp['weight_decay'])
            scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 1.0, total_iters=len(train_loader))

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
            boundaries = torch.tensor([2.5, 2.5, 15/180, 0.1, 0.1, 0.1, 0.1]) # symmetric for each dimension about zero
            adapt_params = exp['adaptation_params']

            samps_list = []
            for sal_idx in range(model.input_transform.nsals):
                samps = torch.zeros(sum(adapt_params.values())) # create empty tensor to hold sampled parameters
                for p_idx, param in enumerate(adapt_params.keys()):
                    if adapt_params[param]: samps[p_idx] = 2*torch.rand(1)-1

                if W > 1:
                    samps[0] = 2*boundaries[0]*samps[0]/(W-1)
                else:
                    samps[0] = 0.0
                if H > 1:
                    samps[1] = 2*boundaries[1]*samps[1]/(H-1)
                else:
                    samps[1] = 0.0
                samps[2] = boundaries[2]*samps[2]
                samps[3] = torch.pow((1 + torch.abs(samps[3])*boundaries[3]), torch.sign(samps[3]) ) # generates scalings appropriately
                samps[4] = torch.pow((1 + torch.abs(samps[4])*boundaries[4]), torch.sign(samps[4]) )
                samps[5] = boundaries[5]*samps[5] # shear
                samps[6] = boundaries[6]*samps[6] # shear
                samps_list.append(samps)
            samps = torch.stack(samps_list, dim=1)
            model.true_params = samps

            # Set-up model to apply input transform
            model.input_transform.mode = 'bicubic'
            model.input_transform.constrain_params = False
            model.input_transform.reset_params(*samps)

            X_test_original = X_test.detach().clone()
            with torch.no_grad():
                X_adapt = model.input_transform(X_adapt)
                X_test = model.input_transform(X_test) # apply affine transformation to test set
            for idx, key in enumerate(true_params.keys()): true_params[key].append(samps[idx, :].detach().cpu().clone().tolist()) # track ground truth params

            # plt.figure()
            # fig, ax = plt.subplots(2, 6)
            # for idx in range(2):
            #     for jdx in range(6):
            #         label = idx*6 + jdx
            #         ax[idx, jdx].imshow(X_test[Y_train==label,0,:,:].mean(dim=0))
            #         ax[idx, jdx].axis('off')
            #         ax[idx, jdx].set_title(f'Label: {label}')
            
            # plt.savefig('baseline-transformed')
            # plt.close()

            # Apply transforms and reload data loaders
            adapt_data = EMGFrameLoader(X=X_adapt, Y=Y_adapt, train=False, norm=exp['norm'], stats=train_data.stats)
            adapt_loader = DataLoader(adapt_data, batch_size=exp['batch_size'], shuffle=True)
            test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
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
            subdists = []
            for sal_idx in range(model.input_transform.nsals):
                subdists.append(get_grid_distance((1,1,H,W), samps[:, sal_idx], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
            dists.append(subdists)
            print('AVERAGE ELECTRODE DISTANCE BETWEEN GRIDS (CM):', f'{dists[-1]} cm')

            # Obtain oracle performance
            adapted_model = deepcopy(model)
            with torch.no_grad():
                X_test_oracle = model.input_transform(X_test, inverse=True)

            test_data_oracle = EMGFrameLoader(X=X_test_oracle, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
            test_loader_oracle = DataLoader(test_data_oracle, batch_size=exp['batch_size'], shuffle=False)
            # Record performance prior to adaptation
            print('TESTING WITH ORACLE OPTIMAL PARAMETERS...')
            adapted_model.eval()
            with torch.no_grad():
                all_labs, all_preds = test_model(adapted_model, test_loader_oracle)

            oracle_acc = accuracy_score(all_labs, all_preds)
            oracle_f1 = f1_score(all_labs, all_preds, average='macro')
            oracle_accs.append(oracle_acc)
            oracle_f1_scores.append(oracle_f1)
            print('Oracle Test Accuracy:', oracle_acc)
            print('Oracle F1 Score:', oracle_f1)

            # Reset SAL parameters
            model.input_transform.constrain_params = True
            adapted_model.input_transform.reset_params()

            # Spatially adapt the model's transformed position
            adapted_model.adaptation_phase = True
            print('FINE-TUNING...')
            for param in adapted_model.parameters():
                param.requires_grad = False
            for param_name in exp['adaptation_params'].keys():
                param = getattr(adapted_model.input_transform, param_name)
                for p in param: p.requires_grad = exp['adaptation_params'][param_name]

            # Aggregate adaptation data per class
            labels = torch.unique(Y_adapt)
            X_adapt_search = torch.zeros(len(labels), 1, X_adapt.shape[2], X_adapt.shape[3], device=device)
            with torch.no_grad():
                for idx, label in enumerate(labels):
                    X_adapt_search[idx, 0, :, :] = torch.sqrt((X_adapt[Y_adapt == label]**2).mean(dim=0))
                    # X_adapt_search = X_adapt[Y_adapt == label].mean(dim=0, keepdim=True)
            adapt_search_data = EMGFrameLoader(X=X_adapt_search, Y=labels, train=False, norm=exp['norm'], stats=train_data.stats)
            adapt_search_loader = DataLoader(adapt_search_data, batch_size=len(labels), shuffle=True)

            print('INITIAL CONDITION SAMPLING...')
            boundaries = torch.tensor([2.5, 2.5, 15/180, 0.1, 0.1, 0.1, 0.1]) # symmetric for each dimension about zero
            initial_search(adapted_model, adapt_search_loader, boundaries, exp['adaptation_params'], H=H, W=W, npoints=int(4**7)) # find optimal initial condition
            adapted_model.input_transform.mode = 'bilinear'

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()),                                                                                
                                lr=exp['lr'], weight_decay=exp['weight_decay'])
            scheduler_params = exp['scheduler']['params']
            scheduler_params['milestones'] = [mlst*data['num_repetitions'] for mlst in scheduler_params['milestones']]
            scheduler = eval(exp['scheduler']['def'])(optimizer, **scheduler_params)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 1.0, total_iters=len(test_loader)*data['num_repetitions'])

            # Adapt to given test set
            train_model(adapted_model, adapt_search_loader, optimizer, criterion, num_epochs=500, scheduler=scheduler,
                        warmup_scheduler=warmup_scheduler, simulation=True) # run training loop

            # Store learned params for later evaluation
            cur_learned_params = []
            for nsal_idx in range(model.input_transform.nsals):
                params = adapted_model.input_transform.get_constrained_params(nsal_idx)
                params = [p.detach().cpu().clone() for p in params]
                cur_learned_params.append(params)
            cur_learned_params = torch.stack([torch.stack(row) for row in cur_learned_params]).T

            for idx, param_name in enumerate(exp['adaptation_params'].keys()):
                learned_params[param_name].append(cur_learned_params[idx,:])
                # cur_learned_params.append(param)
            
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
            subcorrected_dists = []
            for sal_idx in range(model.input_transform.nsals):
                subcorrected_dists.append(get_grid_distance((1,1,H,W), samps[:, sal_idx], cur_learned_params[:, sal_idx]))
            corrected_dists.append(subcorrected_dists)
            print('AVERAGE ELECTRODE DISTANCE BETWEEN GRIDS (CM):', f'{corrected_dists[-1]} cm')

            # Hyser baseline transform
            with torch.no_grad():
                X_test_fixed = adapted_model.input_transform(X_test)
            
            # plt.figure()
            # fig, ax = plt.subplots(2, 6)
            # for idx in range(2):
            #     for jdx in range(6):
            #         label = idx*6 + jdx
            #         ax[idx, jdx].imshow(X_test_fixed[Y_train==label,0,:,:].mean(dim=0))
            #         ax[idx, jdx].axis('off')
            #         ax[idx, jdx].set_title(f'Label: {label}')
            
            # plt.savefig('hyser-baseline-fixed')
            # plt.close()

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

    # Initialize wandb and make sure no other runs are active concurrently for interference
    while wandb.run is not None and not wandb.run._is_finished():
        time.sleep(3)

    # Log wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=exp.pop("project"),
        config=config,
        name=name
        # mode='disabled',
    )

    # Logging final results onto wandb 
    table = wandb.Table(dataframe=df)
    wandb.log({'complete_results': table})
    wandb.log({'Tuned Accuracy': df['Tuned Accuracy'].mean()})


    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
    wandb.log({'Time Ellapsed':f'{h}h, {m}min'})
    wandb.finish()

