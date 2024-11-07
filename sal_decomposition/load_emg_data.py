import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import TensorDataset, DataLoader


def loadmat_decomp(DIR, name, MVC=20):
    '''Loads decomposition data uploaded by Simon'''
    arr = loadmat(os.path.join(DIR, name))

    # Load acquisition specs
    specs = {}
    specs['fs'] = arr['signal'][0,0][2][0,0]
    specs['Nch'] = arr['signal'][0,0][3][0,0]
    specs['Ngrids'] = arr['signal'][0,0][4][0,0]
    specs['path'] = arr['signal'][0,0][7]
    specs['target'] = arr['signal'][0,0][8]
    specs['coords'] = arr['signal'][0,0][9]
    specs['IED'] = arr['signal'][0,0][10]

    # Load data for all grids
    EMG = arr['signal'][0,0][0][64:128,:]
    pulse_trains = arr['signal'][0,0][13][0, 1]

    # Filter data based on target signal
    thrs = MVC-0.2
    isometric = specs['target'].ravel() > thrs
    EMG, pulse_trains = EMG[:, isometric], pulse_trains[:, isometric]
    return EMG, pulse_trains, specs


def get_spike_matrix(pulse_trains, L, K):
    '''Takes pulse trains estimated from data and gives us the spike train matrix used in the decomposition problem.'''
    T = np.zeros((pulse_trains.shape[0]*(L+K-1), pulse_trains.shape[1] - L - K + 1), dtype=float) # extended representation
    print('GETTING SPIKE MATRIX...')
    for tdx in tqdm(range(pulse_trains.shape[1] - L - K + 1)):
        pulse_trains_tdx = np.fliplr(pulse_trains[:, tdx:tdx + L + K - 1])
        T[:, tdx] = pulse_trains_tdx.ravel()
    return T

def get_observation_matrix(EMG, L, K):
    '''Takes observed data and gives us the observation format used in the decomposition problem.'''
    Y = np.zeros((EMG.shape[0]*(K), EMG.shape[1] - L - K + 1), dtype=float) # extended representation, same length as T
    print('GETTING OBSERVATION MATRIX...')
    for tdx in tqdm(range(EMG.shape[1] - L - K + 1)):
        EMG_tdx = EMG[:, tdx:tdx + K]
        Y[:, tdx] = EMG_tdx.ravel()
    return Y

# Regression model

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Define a single linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        # Forward pass: pass input through the linear layer
        return self.linear(x)



if __name__ == '__main__':
    DIR ='/home/joao/Desktop/datasets/simon/ta_grid'
    name = 'S1_20_DF.otb+_decomp.mat_edited.mat'
    EMG, pulse_trains, specs = loadmat_decomp(DIR, name)

    # Decomp. parameters
    N, M, K, L = pulse_trains.shape[0], EMG.shape[0], 40, 30
    pca = PCA(whiten=True)
    T = torch.tensor(get_spike_matrix(pulse_trains, L=L, K=K), dtype=torch.float32).T
    Y = get_observation_matrix(EMG, L=L, K=K).T
    Y = torch.tensor(pca.fit_transform(Y), dtype=torch.float32)
    # Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    
    # Torch data loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TensorDataset(T.to(device), Y.to(device)) # modelling as multivariate linear regression
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True) # get data loader to load in data
    MUAPs = LinearRegressionModel(T.shape[1], Y.shape[1]).to(device)

    # Optimization parameters
    # Define a loss function (Mean Squared Error Loss)
    criterion = torch.nn.MSELoss()

    # Define an optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.Adam(MUAPs.parameters(), lr=0.01, weight_decay=1e-5)

    num_epochs = 20
    losses = []
    for epoch in tqdm(range(num_epochs)):
        for batch_X, batch_Y in dataloader:
            # Move the batch data to GPU
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Compute predicted y by passing x to the model
            predictions = MUAPs(batch_X)

            # Compute the loss
            loss = criterion(predictions, batch_Y)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            losses.append(loss.detach().cpu().numpy())

            # Perform a single optimization step (parameter update)
            optimizer.step()

        # Print progress
        # if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    # Print the training loss
    plt.figure()
    plt.plot(losses)
    plt.title('Running Training Loss')
    plt.show()

    # Get learned mixing matrix and see what we can extract from it
    H = MUAPs.state_dict()['linear.weight'].cpu().T    #
    # plt.figure()
    # plt.plot(np.convolve(H[:, 0], np.ones(50)/50, mode='same'), color='orange')
    # plt.plot(H[:, 0])
    # plt.vlines(np.arange(1, pulse_trains.shape[0])*(L+K-1), -1, 1, color='red')
    # plt.show()
    for idx in range(15):
        fig, axs = plt.subplots(2)
        axs[0].plot(np.convolve(H[:, idx], np.ones(50)/50, mode='same'))
        axs[1].plot(H[:, idx])
        axs[0].vlines(np.arange(1, pulse_trains.shape[0])*(L+K-1), -1, 1, color='red')
        axs[1].vlines(np.arange(1, pulse_trains.shape[0])*(L+K-1), -1, 1, color='red')
        plt.show()
    # Make dummy image to investigate 
    # img = np.mean(EMG)*np.ones((13,5))
    # acc = 64
    # for cdx in range(5):
    #     if cdx % 2 != 0: # revert order when coming back
    #         rows = range(0,13)
    #     else:
    #         rows = range(12,-1,-1)
    #     for rdx in rows:
    #         if not (cdx==0 and rdx==0):
    #             stat = np.std(EMG[acc, :])
    #             img[rdx, cdx] = stat
    #             rdx, cdx
    #             acc = acc + 1
    
    # plt.figure()
    # plt.imshow(gaussian_filter(img, sigma=1))
    # plt.colorbar()
    # plt.show()
