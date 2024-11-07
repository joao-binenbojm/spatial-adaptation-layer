import numpy as np
import pickle
# from MUEdit import emg_decomposition_final, processing_tools

# def Hdims(IED, l, N=100, K=20, Lsamp=10):
#     ''' Compute mixing matrix dimensions for simulation purposes.'''
#     return int( ((l**2)/(3*IED**2)) + (4*l/IED) + 1 )*K, N*(Lsamp + K - 1)

# print(Hdims(0.01, 29, K=1), np.prod(Hdims(0.01, 29, K=1)))

if __name__ == '__main__':
    with open('./sal_decomposition/decomposition_data.pkl', 'rb') as f:
        decomp = pickle.load(f)
    print()