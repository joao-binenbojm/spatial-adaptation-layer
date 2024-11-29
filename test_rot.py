import numpy as np
import matplotlib.pyplot as plt
# from PIL.Image import rotate
from scipy.ndimage import rotate
import torch

def apply_affine(emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, sampfactor=100):
    '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''

    N, C, H, W = emg_grid.shape
    Tx, Ty = torch.tensor(2*Tx*sampfactor/W), torch.tensor(2*sampfactor*Ty/H) # Normalize translation values automatically
    theta, xscale, yscale = torch.tensor(theta), torch.tensor(xscale), torch.tensor(yscale)

    T = torch.cat([ # Translation Matrix
        torch.stack([torch.tensor(1.0), torch.tensor(0.0), Tx]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(1.0), Ty]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)
    R = torch.cat([ # Rotation Matrix
        torch.stack([torch.cos(theta), -torch.sin(theta), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.sin(theta), torch.cos(theta), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)
    Sc = torch.cat([ # Scaling Matrix
        torch.stack([xscale, torch.tensor(0.0), torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), yscale, torch.tensor(0.0)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
    ], dim=0)

    theta = Sc @ R @ T
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(N,1,1)
    grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=False)
    xresamp = torch.nn.functional.grid_sample(emg_grid, grid)
    
    return xresamp

theta = np.pi*10/180
W = 10
H = 24
xcrop = (W*(np.cos(theta) - 1) + H*np.sin(theta))/(2*(np.cos(theta) + np.sin(theta)))
xcrop = int(np.ceil(xcrop))
ycrop = xcrop

im = np.zeros((H,W))
im[ycrop:H-ycrop, xcrop:W-xcrop] = np.ones((H-2*ycrop, W - 2*xcrop))

plt.figure()
plt.imshow(im)
plt.show()

plt.figure()
plt.imshow(rotate(im, angle=10, reshape=False))
plt.show()

