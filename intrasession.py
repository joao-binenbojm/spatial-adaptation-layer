import os
from time import time
import json
import gc

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


# from data_loaders import load_tensors, extract_frames_csl, extract_frames_capgmyo, EMGFrameLoader
from tensorize_emg import CapgmyoData, CSLData, CapgmyoDataRMS, CSLDataRMS
from torch_loaders import EMGFrameLoader
from utils.deep_learning import train_model, test_model
from utils.emg_processing import majority_voting_segments, majority_voting_capgmyo
from networks import CapgMyoNet, CapgMyoNetInterpolate, RMSNet
from networks import median_pool_2d

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
    subs, sessions, test_reps = [], [], []
    accs, maj_accs, maj_accs_capgmyo = [], [], [] # different metrics to be saved in csv from experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 

    print('INTRASESSION:', data['dataset_name'])
    for idx, sub in tqdm(enumerate(data['subs'])):
        # Load data for given subject/session
        # dg = data['dgs'][idx]
        sub_id = 'subject{}'.format(sub+1)

        # Load EMG data in uniform format
        emg_tensorizer = emg_tensorizer_def(path=data['DIR'], sub=sub_id, num_gestures=data['num_gestures'], num_repetitions=data['num_repetitions'],
                                            input_shape=data['input_shape'], fs=data['fs'], sessions=session_ids, intrasession=True)
        emg_tensorizer.load_tensors()

        for session in tqdm(data['sessions']):
            for test_idx in range(10):
                subs.append(sub)
                sessions.append(session)
                test_reps.append(test_idx)
                print('\n SUBJECT #{}, SESSION #{}'.format(sub + 1, session + 1))
                print('TEST REPETITION #{}'.format(test_idx + 1))

                X_train, Y_train, X_test, Y_test = emg_tensorizer.get_tensors(test_session=session, rep_idx=test_idx)

                # COMPUTE THE AVERAGE EMG IMAGE FOR EACH GESTURE
                X_train_plot = median_pool_2d(X_train)
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

                # # plt.figure()
                # # plt.imshow(meanX[25, :, :], cmap='gray')
                # # plt.title('25')
                # # plt.savefig('25.jpg')
                
                # # plt.figure()
                # # plt.imshow(meanX[26, :, :], cmap='gray')
                # # plt.title('26')
                # # plt.savefig('26.jpg')
                
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
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=exp['lr'], momentum=exp['momentum'], weight_decay=exp['weight_decay'])
                scheduler = eval(exp['scheduler']['def'])(optimizer, **exp['scheduler']['params'])
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=len(train_loader))

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

                # Majority Voting Accuracy
                maj_all_preds = majority_voting_segments(all_preds, M=1000, n_samples=2048)
                maj_acc = accuracy_score(all_labs, maj_all_preds)
                maj_accs.append(maj_acc)
                print('Majority Voting Accuracy:', maj_acc)

                # Majority Voting Capgmyo Style
                maj_all_preds2, maj_all_labs = majority_voting_capgmyo(all_preds, n_samples=2048), majority_voting_capgmyo(all_labs, n_samples=2048)
                maj_acc_capgmyo = accuracy_score(maj_all_labs, maj_all_preds2)
                maj_accs_capgmyo.append(maj_acc_capgmyo)
                print('Majority Voting Accuracy (A la Capgmyo):', maj_acc_capgmyo)


                # Plotting confusion matrix to understand what's going on
                labs = np.arange(1, 27)
                cf = confusion_matrix(maj_all_labs, maj_all_preds2, labels=labs)
                disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labs)
                disp.plot()
                plt.savefig('cfm.jpg')
                plt.close()

                # Plotting a prediction-label stream
                plt.figure()
                plt.plot(all_labs)
                plt.plot(maj_all_preds)
                plt.legend(['Labels', 'Predictions'])
                plt.savefig('stream_maj.jpg')
                plt.close()

                plt.figure()
                plt.plot(all_labs)
                plt.plot(all_preds)
                plt.legend(['Labels', 'Predictions'])
                plt.savefig('stream.jpg')
                plt.close()

                # SAVE RESULTS
                arr = np.array([subs, sessions, test_reps, accs, maj_accs, maj_accs_capgmyo]).T
                df = pd.DataFrame(data=arr, columns=['Subjects', 'Sessions', 'Test Repetitions', 'Accuracy', 'Majority Voting Accuracy', 'Majority Voting Accuracy (Capgmyo)'])
                df.to_csv(name + '.csv')

    # Save experiment data in .csv file
    arr = np.array([subs, sessions, test_reps, accs, maj_accs, maj_accs_capgmyo]).T
    df = pd.DataFrame(data=arr, columns=['Subjects', 'Sessions', 'Test Repetitions', 'Accuracy', 'Majority Voting Accuracy', 'Majority Voting Accuracy (Capgmyo)'])
    df.to_csv(name + '.csv')

    tf = time()
    h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
    print('EXPERIMENT #{} - TOTAL TIME ELAPSED: {}h, {}min'.format(name, h, m))
