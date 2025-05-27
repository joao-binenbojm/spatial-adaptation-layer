import torch
import scipy
from time import time
import matplotlib.pyplot as plt
import wandb
from sal_classification.simulation_utils import get_grid_distance
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def init_adabn(model):
    '''Takes a given PyTorch model, sets all modules to evaluation mode, then resets BN statistics
        and sets BN modules to training mode. This ensures that the next forward pass updates the 
        running statistics tracked.
    '''
    model.eval()
    for name, module in model.named_modules():
        if ('batch_norm' in name) or ('bn' in name) or ('batchnorm' in name): # if a BN module
            module.reset_running_stats() # resets mean/std and batch counter
            module.train() # ensures that stats are updated in the following forward pass
            module.momentum = None # keep track of simple cumulative mean

## TRAINING/TESTING
def add_noise_input_transform(model, std=0.01):
    with torch.no_grad():
        for param in model.input_transform.parameters():
            noise = torch.randn_like(param) * std
            param.add_(noise)

def get_inv_constrained_params(*args, boundaries):
    '''Maps parameters in constrained search space back to unconstrained space using the inverse of the sigmoid function.'''
    args_out = list(args).copy()
    for arg_idx, arg in enumerate(args):
        if boundaries[arg_idx][1] != boundaries[arg_idx][0]:
            args_out[arg_idx] = torch.log((arg - boundaries[arg_idx][0]) / (boundaries[arg_idx][1] - arg))
        else:
            args_out[arg_idx] = boundaries[arg_idx][1] # accounts for when parameter is not searched for

    return args_out

def initial_search(model, train_loader, boundaries, adaptation_params, H, W, npoints=50):
    ''' Sample N initial spatial transformations and choose optimal as starting point'''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on
    criterion = torch.nn.CrossEntropyLoss() # loss function
    
    # Searching through initial conditions
    print('SAMPLING AND EVALUATING INITIAL CONDITIONS...')
    d = sum(adaptation_params.values())
    
    # Latin hypercube sampling
    engine = scipy.stats.qmc.LatinHypercube(d=d)
    samp_params = 2*torch.tensor(engine.random(n=npoints)).to(torch.float32)-1 # scale from [0,1] to [-1, 1]    
    
    # Appropriately scale sampled parameters that are included in
    init_params = torch.zeros((npoints, len(adaptation_params)))
    param_count = 0
    for p_idx, param in enumerate(adaptation_params.keys()):
        if adaptation_params[param]:
            init_params[:, p_idx] = samp_params[:, param_count]
            param_count += 1
    
    if W > 1:
        init_params[:,0] = 2*boundaries[0]*init_params[:,0]/(W-1)
    else:
        init_params[:,0] = 0.0
    if H > 1:
        init_params[:,1] = 2*boundaries[1]*init_params[:,1]/(H-1)
    else:
        init_params[:,1] = 0.0
    init_params[:, 2] = boundaries[2]*init_params[:, 2] # rotation
    init_params[:, 3] = torch.pow((1 + torch.abs(init_params[:, 3])*boundaries[3]), torch.sign(init_params[:, 3]) ) # generates scalings appropriately
    init_params[:, 4] = torch.pow((1 + torch.abs(init_params[:, 4])*boundaries[4]), torch.sign(init_params[:, 4]) )
    init_params[:, 5] = boundaries[5]*init_params[:, 5]
    init_params[:, 6] = boundaries[6]*init_params[:, 6] # shear

    # Identity parameters: include in case no transformation is optimal
    identity_params = torch.zeros((1, len(adaptation_params)))
    identity_params[:,3:5] = 1.0 # xscale, yscale
    init_params = torch.cat((init_params, identity_params), dim=0) # add identity parameters to search space
    npoints += 1 # account for identity parameters
    losses = torch.zeros(model.input_transform.nsals, npoints)

    init_params = init_params.to(device)
    model.input_transform.constrain_params = False # disable parameter constraints for initial search
    with torch.no_grad():
        for sal_idx in range(model.input_transform.nsals):
            print(f'Spatial Adaptation Layer #{sal_idx+1}...')
            for npoint in tqdm(range(npoints)):
                # Set initial conditions
                model.input_transform.xshift[sal_idx].copy_(init_params[npoint, 0])
                model.input_transform.yshift[sal_idx].copy_(init_params[npoint, 1])
                model.input_transform.rot_theta[sal_idx].copy_(init_params[npoint, 2])
                model.input_transform.xscale[sal_idx].copy_(init_params[npoint, 3])
                model.input_transform.yscale[sal_idx].copy_(init_params[npoint, 4])
                model.input_transform.xshear[sal_idx].copy_(init_params[npoint, 5])
                model.input_transform.yshear[sal_idx].copy_(init_params[npoint, 6])

                # Get batch estimate of supervised loss
                total_loss = 0
                for i, (signals, labels) in enumerate(train_loader):
                    signals = signals.to(device)
                    labels = labels.view(-1).type(torch.LongTensor).to(device)
                    # forward pass
                    outputs = model(signals).to(device)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                losses[sal_idx, npoint] = total_loss

            if npoints > 0:
                best_params = init_params[losses[sal_idx].argmin(), :]
                print('BEST PARAMS:', best_params)

                # Map from constrained to unconstrained space before training
                best_params = get_inv_constrained_params(*best_params, boundaries=model.input_transform.boundaries)
                model.input_transform.xshift[sal_idx].copy_(best_params[0])
                model.input_transform.yshift[sal_idx].copy_(best_params[1])
                model.input_transform.rot_theta[sal_idx].copy_(best_params[2])
                model.input_transform.xscale[sal_idx].copy_(best_params[3])
                model.input_transform.yscale[sal_idx].copy_(best_params[4])
                model.input_transform.xshear[sal_idx].copy_(best_params[5])
                model.input_transform.yshear[sal_idx].copy_(best_params[6])
    model.input_transform.constrain_params = True # re-enable parameter constraints


def train_model(model, train_loader, optimizer, criterion, num_epochs=2, scheduler=None, warmup_scheduler=None, val_loader=None, val_acc_threshold=0.8, verbose=True, simulation=False):
    '''Training loop for given experiment.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
    running_correct = 0
    xshift, yshift, baseline = [], [], []
    xshift2, yshift2 = [], []
    dists = []
    weights = []
    running_losses = []
    val_losses = [float('inf')]
    n_restarts = 0
    model_tracker = {"model": None, "val_loss": float('inf')}

    t0 = time() # initial timestamp at start of training
    epoch = 0
    while epoch < num_epochs:
    # for epoch in range(num_epochs):
        if verbose:
            print('Learning Rate:', scheduler.get_last_lr())
        running_loss = 0.0
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.view(-1).type(torch.LongTensor).to(device)
            # forward pass
            outputs = model(signals).to(device)
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()

            # TENSORBOARD
            running_loss += loss.item()
            baseline.append(model.baseline.cpu().detach().numpy().ravel())

            # if 'spatial' in model.input_transform.name:
            #     xshift.append(model.input_transform.xshift.cpu().detach().numpy())
            #     yshift.append(model.input_transform.yshift.cpu().detach().numpy())

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted.squeeze() == labels.view(-1)).sum().item()

            if simulation:
                # Extract parameters and compute current distance
                for param_name in ['xshift', 'yshift', 'rot_theta', 'xscale', 'yscale','xshear', 'yshear']:
                    cur_learned_params = []
                    param = getattr(model.input_transform, param_name, None)
                    if param: cur_learned_params.append(param[0].item())
                # dists.append(get_grid_distance(signals[[0],:,:,:].shape, model.true_params, cur_learned_params))

            if (i + 1) % 20 == 0:
                if verbose:
                    print('Epoch {} / {}, step {} / {}, training loss = {:4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                running_losses.append(running_loss)
                running_loss = 0.0
                running_correct = 0
        epoch += 1

        # if val_loader:
        #     with torch.no_grad():
        #         val_loss = 0.0
        #         for i, (signals, labels) in enumerate(val_loader):
        #             signals = signals.to(device)
        #             labels = labels.view(-1).type(torch.LongTensor).to(device)
        #             # Forward pass
        #             outputs = model(signals).to(device)
        #             loss = criterion(outputs, labels)
        #             val_loss += loss.item()
        #         val_loss = val_loss / len(val_loader)

        #         if val_loss < model_tracker["val_loss"]:
        #             model_tracker["model"] = model.state_dict()
        #             model_tracker["val_loss"] = val_loss
        #             print('Model saved with validation loss: {}'.format(val_loss))

        #         # Whether convergence was achieved at the appropriate model 
        #         if val_loss >  min(val_losses) - 1e-6:
        #             all_labs, all_preds = test_model(model, val_loader)
        #             val_acc = accuracy_score(all_labs, all_preds)
        #             if val_acc > val_acc_threshold:
        #                 print('Convergence achieved at epoch {} with accuracy {}'.format(epoch, val_acc))
        #                 if model_tracker["model"]:
        #                     model.load_state_dict(model_tracker["model"])
        #                 break
        #             else:
        #                 if n_restarts < 5:
        #                     print('Model stuck at epoch {} with accuracy {}. \n Resetting...'.format(epoch, val_acc))
        #                     epoch = 0
        #                     val_losses = [1e10]
        #                     model.input_transform.restart()
        #                     n_restarts += 1
        #                 else:
        #                     print('5 Restarts reached. \n Finish adaptation..')
        #                     if model_tracker["model"]:
        #                         model.load_state_dict(model_tracker["model"])
        #                     break
                
        #         else:
        #             val_losses.append(val_loss)
        #             print('Epoch {} / {}, validation loss = {:4f}'.format(epoch, num_epochs, val_loss))
        #             # writer.add_scalar('validation loss', val_loss, epoch * len(train_loader) + i)
        #             # writer.add_scalar('validation accuracy', val_acc, epoch * len(train_loader) + i)
        #             # if train: wandb.log({'Validation Loss': val_loss})
        #             # else: wandb.log({'Fine-tuning Loss': val_loss})
                

        # Update scheduler and calculate time taken after given epoch
        scheduler.step()
        tf = time()
        h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
        print('TOTAL TIME ELAPSED: {}h, {}min'.format(h, m))
    
    # Plot learnable shifts and baseline
    # xshift, yshift = 
    # if 'model.shift' in locals():
    # plt.figure()
    # plt.plot(xshift)
    # plt.plot(yshift)
    # if 'spatial_adapt1' in dir(model):
    #     plt.plot(xshift2)
    #     plt.plot(yshift2)
    # plt.legend(['xshift', 'yshift', 'xshift2', 'yshift2'])
    # plt.savefig('shifts.jpg')
    # plt.close()

    # Plot grid distance dynamics
    plt.figure()
    plt.plot(dists)
    plt.title('Grid distance dynamics')
    plt.savefig('grid_distance.jpg')
    plt.close()

    # if 'model.baseline' in locals():
    plt.figure()
    plt.plot(baseline)
    plt.title('Learned baseline')
    plt.savefig('baseline.jpg')
    plt.close()

    # plt.figure()
    # plt.plot(weights)
    # plt.title('Learned weights')
    # plt.savefig('weights{}.jpg'.format(optimizer.param_groups[0]['lr']))
    # plt.close()

    plt.figure()
    plt.plot(running_losses)
    plt.title('TRAINING LOSS')
    plt.savefig('loss.jpg')
    plt.close()

def test_model(model, test_loader):
    ''' Takes given PyTorch model and test DataLoader, and returns all labels and corresponding model predictions.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
    all_labs, all_preds = [], []
    for i, (signals, labels) in enumerate(test_loader):
        signals, labels = signals.to(device), labels.view(-1).to(device)
        outputs = model(signals).to(device)
        _,predictions = torch.max(outputs, 1) # get class labels
        all_labs.extend(labels.cpu().tolist())
        all_preds.extend(predictions.cpu().tolist())
    return all_labs, all_preds
