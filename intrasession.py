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
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt


# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS
from torch_loaders import EMGFrameLoader
from utils.deep_learning import train_model, test_model
from networks import CapgMyoNet, CapgMyoNetInterpolate, RMSNet

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/capgmyo')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

if __name__ == '__main__':

    exp = 'exp.json'

    # Experiment condition loading
    name = exp[:-5] # keep experiment name
    print('#'*40 + '\n\n' + 'EXPERIMENT:'+ name + '\n\n' + '#'*40)

    with open('capgmyo.json') as f:
        data = json.load(f)
    with open(exp) as f:
        exp = json.load(f)
    emg_tensorizer_def = eval(exp['emg_tensorizer'])

    t0 = time()

    # Preinitialize metric arrays
    session_ids = ['session'+str(ses) for ses in data['sessions']]
    subs, test_reps = [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('INTRASESSION:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        # Load data for given subject/session
        dg = data['dgs'][idx]
        sub_id = 'subject{}'.format(sub)

        # Load EMG data in uniform format
        emg_tensorizer = emg_tensorizer_def(path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=session_ids, 
                                            median_filt=exp['median_filt'], intrasession=exp['intrasession'])
        emg_tensorizer.load_tensors()

        for test_ses in tqdm(data['sessions']):
            for test_idx in range(10):
                subs.append(sub)
                test_reps.append(test_idx)
                print('\n SUBJECT #{}'.format(sub))
                print('TEST REPETITION #{}, dg: {}'.format(test_idx, dg))

                X_train, Y_train, X_test, Y_test = emg_tensorizer.get_tensors(test_idx=test_ses-1, dg=dg)

                # # COMPUTE THE AVERAGE EMG IMAGE FOR EACH GESTURE
                # fullX = data_extractor.X[0]
                # meanX = fullX.mean(dim=(1,2,3))
                # fig, axs = plt.subplots(2, 4)
                # for idx in range(2):
                #     for jdx in range(4):
                #         axs[idx, jdx].imshow(meanX[idx*4 + jdx,:,:], cmap='gray')
                #         axs[idx, jdx].set_title(str(idx*4 + jdx))
                # plt.show()
                
                # Get PyTorch DataLoaders
                train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'])
                test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=True)

                # Model/training set-up
                Nv, Nh = emg_tensorizer.Nv, emg_tensorizer.Nh
                model = eval(exp['network'])(channels=np.prod((Nv, Nh)), input_shape=(Nv, Nh), num_classes=data['num_gestures']).to(device)
                num_epochs = exp['num_epochs']
                criterion = nn.CrossEntropyLoss(reduction='sum')
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=3*len(train_loader))

                # Train the model
                if exp['adaptation'] == 'shift-adaptation':
                    model.shift.xshift.requires_grad = False
                    model.shift.yshift.requires_grad = False
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

                # SAVE RESULTS
                arr = np.array([subs, test_reps, accs]).T
                df = pd.DataFrame(data=arr, columns=['Subjects', 'Test Repetitions', 'Accuracy'])
                df.to_csv(name + '.csv')

    # Save experiment data in .csv file
    arr = np.array([subs, test_reps, accs]).T
    df = pd.DataFrame(data=arr, columns=['Subjects', 'Test Repetitions', 'Accuracy'])
    df.to_csv(name + '.csv')

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
