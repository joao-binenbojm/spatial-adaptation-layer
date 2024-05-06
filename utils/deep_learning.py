import torch
from time import time
import matplotlib.pyplot as plt


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
def train_model(model, train_loader, optimizer, criterion, num_epochs=2, scheduler=None, warmup_scheduler=None, val_loader=None):
    '''Training loop for given experiment.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # choose device to let model training happen on 
    running_correct = 0
    xshift, yshift, baseline = [], [], []
    weights = []
    running_losses = []

    t0 = time() # initial timestamp at start of training
    for epoch in range(num_epochs):
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

            # weights.append(model.fc.weight.cpu().detach().numpy().ravel())
#            baseline.append(model.baseline.cpu().detach().numpy().ravel())

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted.squeeze() == labels.view(-1)).sum().item()

            if (i + 1) % 20 == 0:
                print('Epoch {} / {}, step {} / {}, loss = {:4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                # writer.add_scalar('training loss', running_loss/100, epoch * len(train_loader) + i)
                # writer.add_scalar('training accuracy', running_correct/100, epoch * len(train_loader) + i)
                running_losses.append(running_loss)
                running_loss = 0.0
                running_correct = 0
        # Update scheduler and calculate time taken after given epoch
        scheduler.step()
        tf = time()
        h, m = ((tf - t0) / 60) // 60, ((tf - t0) / 60) % 60
        print('TOTAL TIME ELAPSED: {}h, {}min'.format(h, m))
    
    # # Plot learnable shifts
    # plt.figure()
    # plt.plot(xshift)
    # plt.plot(yshift)
    # plt.legend(['xshift', 'yshift'])
    # plt.savefig('shifts{}.jpg'.format(optimizer.param_groups[0]['lr']))
    # plt.close()

 #   plt.figure()
  #  plt.plot(baseline)
   # plt.title('Learned baseline')
    #plt.savefig('baseline.jpg')
   # plt.close()
    
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
