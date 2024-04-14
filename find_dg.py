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

from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, EMGFrameLoader
from utils.emg_processing import majority_voting_segments, 
from utils.deep_learning import train_model, test_model, init_adabn
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
    # data_extractor_def = CapgmyoData if data['dataset_name'] == 'capgmyo' else CSLData
    data_extractor_def = CapgmyoDataRMS

    t0 = time()

    # Preinitialize metric arrays
    sessions = ['session'+str(ses) for ses in data['sessions']]
    subs, test_ses, dgs, runs = [], [], [], []
    xshifts, yshifts = [], []
    accs, maj_accs = [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('INTERSESSION:', data['dataset_name'])
    for sub in tqdm(data['subs']):
        # Load data for given subject/session
        print('\n SUBJECT #{}'.format(sub))
        sub_id = 'subject{}'.format(sub)

        # Load EMG data in uniform format
        data_extractor = data_extractor_def(path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=sessions, 
                                            median_filt=exp['median_filt'], baseline=exp['baseline'])
        data_extractor.load_tensors()

        for ses in tqdm(data['sessions']):
            for dg in np.arange(40, 80)/2:
                print('SESSION #{}, dg: {}'.format(ses, dg))

                X_train, Y_train, X_test, Y_test = data_extractor.get_tensors(test_idx=ses-1, dg=dg)

                # # COMPUTE THE AVERAGE EMG IMAGE FOR EACH GESTURE
                # fullX = data_extractor.X[0]
                # meanX = fullX.mean(dim=(1,2,3))
                # fig, axs = plt.subplots(2, 4)
                # for idx in range(2):
                #     for jdx in range(4):
                #         axs[idx, jdx].imshow(meanX[idx*4 + jdx,:,:], cmap='gray')
                #         axs[idx, jdx].set_title(str(idx*4 + jdx))
                # plt.show()
        
                # Translation transforms
                shift = exp['translation_shift'] # in terms of electrodes
                a, b = 0,0 #shift/data['input_shape'][1], shift/data['input_shape'][0]
                transforms = Compose([
                    RandomAffine(degrees=0, translate=(a,b), interpolation=InterpolationMode.BILINEAR)
                ])
                
                # FOR THE APPROPRIATE DATA, TRAIN 5 MODELS
                for run in range(1, 6):
                    print('RUN #{}'.format(run))
                    runs.append(run)
                    subs.append(sub)
                    test_ses.append(ses)
                    dgs.append(dg)
                    # Get PyTorch DataLoaders
                    train_data = EMGFrameLoader(X=X_train, Y=Y_train, norm=exp['norm'], transform=transforms)
                    test_data = EMGFrameLoader(X=X_test, Y=Y_test, train=False, norm=exp['norm'], stats=train_data.stats)
                    train_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True)
                    test_loader = DataLoader(test_data, batch_size=exp['batch_size'], shuffle=True)

                    # Model/training set-up
                    Nv, Nh = data_extractor.Nv, data_extractor.Nh
                    # model = CapgMyoNetInterpolate(channels=np.prod((Nv, Nh)), input_shape=(Nv, Nh), num_classes=data['num_gestures']).to(device)
                    # model = CapgMyoNet(channels=np.prod((Nv, Nh)), input_shape=(Nv, Nh), num_classes=data['num_gestures']).to(device)
                    model = RMSNet(channels=np.prod((Nv, Nh)), input_shape=(Nv, Nh), num_classes=data['num_gestures']).to(device)
                    num_epochs = exp['num_epochs']
                    criterion = nn.CrossEntropyLoss(reduction='sum')
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                                lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=3*len(train_loader))

                    # Train the model
                    model.resample.xshift.requires_grad = False
                    model.resample.yshift.requires_grad = False
                    model.baseline.requires_grad = False
                    train_model(model, train_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop

                    # Fine-tune to update model's shifting position
                    print('FINE-TUNING...')
                    for param in model.parameters():
                        param.requires_grad = False
                    model.eval()
                    model.resample.xshift.requires_grad = True
                    model.resample.yshift.requires_grad = True
                    model.baseline.requires_grad = True
                    for g in optimizer.param_groups:
                        g['lr'] = exp['lr']
                    scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=3*len(test_loader))
                    train_model(model, test_loader, optimizer, criterion, num_epochs=exp['num_epochs'], scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler) # run training loop


                    xshift = model.resample.xshift.cpu().detach().numpy()[0]
                    yshift = model.resample.yshift.cpu().detach().numpy()[0]

                    xshifts.append(xshift)
                    yshifts.append(yshift)

                    # # If AdaBatch, initialize it
                    # if exp['calibrate']:
                    #     print('UPDATING BN STATS...')
                    #     init_adabn(model)
                    #     for i, (signals, labels) in enumerate(test_loader):
                    #         signals = signals.to(device)
                    #         model(signals).to(device) # update batch stats
                        # # If AdaBatch, initialize it
                    # if exp['calibrate']:
                    #     print('UPDATING BN STATS...')
                    #     init_adabn(model)
                    #     for i, (signals, labels) in enumerate(test_loader):
                    #         signals = signals.to(device)
                    #         model(signals).to(device) # update batch stats

                    # Testing loop over test loader
                    print('TESTING...')
                    model.eval()
                    with torch.no_grad():
                        print('LEARNED SHIFTS: x: {} , y: {}'.format(xshift, yshift))
                        all_labs, all_preds = test_model(model, test_loader)

                    acc = accuracy_score(all_labs, all_preds)
                    accs.append(acc)
                    print('Test Accuracy:', acc)

                    # SAVE RESULTS
                    arr = np.array([subs, test_ses, accs, dgs, xshifts, yshifts, runs]).T
                    df = pd.DataFrame(data=arr, columns=['Subjects', 'Test Session', 'Accuracy', 'Electrode Distance', 'xshift', 'yshift', 'Runs'])
                    df.to_csv(name + '.csv')

    # Save experiment data in .csv file
    data = np.array([subs, test_ses, accs, maj_accs, dgs, xshifts, yshifts]).T
    df = pd.DataFrame(data=data, columns=['Subjects', 'Test Session', 'Accuracy', 'Electrode Distance', 'xshift', 'yshift', 'Runs'])
    df.to_csv(name + '.csv')

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
