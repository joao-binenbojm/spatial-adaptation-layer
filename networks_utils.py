import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math
import random

def wrap_grid_horizontally(grid):
    """
    Given a grid of shape (B, H, W, 2), wrap the x coordinates using modulo.
    Assumes x is the 2nd channel in the last dim and is in [-1, 1] range.
    """
    x = grid[..., 0]
    y = grid[..., 1]

    # Normalize to [0, 1], apply modulo, then back to [-1, 1]
    x_wrapped = ((x + 1) / 2) % 1.0  # now in [0,1]
    x_wrapped = x_wrapped * 2 - 1    # back to [-1,1]

    return torch.stack([x_wrapped, y], dim=-1)

def get_P(H, W, theta):
    """
    Build dense bilinear interpolation matrix P for HxW grid given affine theta.
    P @ y_flat maps the EMG grid y into the transformed grid.
    """
    M = H * W
    # Generate normalized affine sampling grid
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), size=(1, 1, H, W), align_corners=True)[0]  # H x W x 2

    # Convert normalized coords to pixel indices
    x = ((grid[..., 0] + 1) * (W - 1) / 2).flatten()
    y = ((grid[..., 1] + 1) * (H - 1) / 2).flatten()

    # Neighbors
    x0 = torch.floor(x).long().clamp(0, W - 1)
    x1 = torch.ceil(x).long().clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    y1 = torch.ceil(y).long().clamp(0, H - 1)

    # Weights
    dx = x - x0.float()
    dy = y - y0.float()

    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    def idx(y_idx, x_idx):
        return y_idx * W + x_idx

    i00 = idx(y0, x0)
    i01 = idx(y0, x1)
    i10 = idx(y1, x0)
    i11 = idx(y1, x1)

    # Assemble dense P
    P = torch.zeros((M, M), dtype=torch.float32, device=theta.device)
    rows = torch.arange(M, device=theta.device)

    P[rows, i00] += w00
    P[rows, i01] += w01
    P[rows, i10] += w10
    P[rows, i11] += w11

    return P

# Precursor of the spatial adaptation layer
# class Shift(torch.nn.Module):
#     def __init__(self, input_shape):
#         super().__init__()
#         self.Nv, self.Nh = input_shape
#         self.xshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
#         self.yshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
#         self.register_buffer('yreg', torch.arange(self.Nv)) # original coordinates
#         self.register_buffer('xreg', torch.arange(self.Nh)) # original coordinates

#     def forward(self, x):
#         '''Regrids input image based on shift parameters.'''
#         yreg = self.yreg - self.yshift
#         xreg = self.xreg - self.xshift
#         H, W = self.Nv, self.Nh
#         xreg, yreg = 2*xreg/(W) - 1, 2*yreg/(H) - 1 # scale between -1 and 1
#         grid_y, grid_x = torch.meshgrid(yreg, xreg, indexing='ij')
#         grid = torch.stack([grid_x, grid_y], dim=-1).view(1, self.Nv, self.Nh, 2)
#         grid = grid.repeat(x.shape[0], 1, 1, 1) # get grid to match batch size dimensions
#         xresamp = torch.nn.functional.grid_sample(x, grid)
#         return xresamp


class SpatialAdaptation(torch.nn.Module):
    def __init__(self, input_shape, T = True, R = True, Sc = True, Sh = True, mode='bilinear', circular=False, boundaries=None, constrain_params=True):
        super().__init__()
        self.input_shape = input_shape
        self.mode = mode
        self.circular = circular
        self.boundaries = boundaries # list of tuples (min, max) for each parameter
        self.constrain_params = constrain_params
        self.nsals = 1
        # Initialize parameters
        self.xshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T)]) 
        self.yshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T)])
        self.rot_theta = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=R)]) # theta in rads.
        self.xscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc)])
        self.yscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc)])
        self.xshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh)])
        self.yshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh)])

    def get_constrained_params(self, sal_idx=0):
        '''Returns the parameters of the affine transformation constrained to the input boundaries using a tanh function.'''
        # tanh outputs in [-1, 1], so map to [min, max] as: 0.5 * (tanh + 1) * (max - min) + min
        xshift = 0.5 * (torch.tanh(self.xshift[sal_idx]) + 1) * (self.boundaries[0][1] - self.boundaries[0][0]) + self.boundaries[0][0]
        yshift = 0.5 * (torch.tanh(self.yshift[sal_idx]) + 1) * (self.boundaries[1][1] - self.boundaries[1][0]) + self.boundaries[1][0]
        rot_theta = 0.5 * (torch.tanh(self.rot_theta[sal_idx]) + 1) * (self.boundaries[2][1] - self.boundaries[2][0]) + self.boundaries[2][0]
        xscale = 0.5 * (torch.tanh(self.xscale[sal_idx]) + 1) * (self.boundaries[3][1] - self.boundaries[3][0]) + self.boundaries[3][0]
        yscale = 0.5 * (torch.tanh(self.yscale[sal_idx]) + 1) * (self.boundaries[4][1] - self.boundaries[4][0]) + self.boundaries[4][0]
        xshear = 0.5 * (torch.tanh(self.xshear[sal_idx]) + 1) * (self.boundaries[5][1] - self.boundaries[5][0]) + self.boundaries[5][0]
        yshear = 0.5 * (torch.tanh(self.yshear[sal_idx]) + 1) * (self.boundaries[6][1] - self.boundaries[6][0]) + self.boundaries[6][0]

        return xshift, yshift, rot_theta, xscale, yscale, xshear, yshear

    def get_affine_transform(self, sal_idx=0, inverse=False):
        '''Returns the affine transformation matrix given the current model parameters.'''
        H, W = self.input_shape
        dev = self.xshift[sal_idx].device
        # Apply soft constraints to parameters
        if self.boundaries and self.constrain_params:
            xshift, yshift, rot_theta, xscale, yscale, xshear, yshear = self.get_constrained_params(sal_idx=sal_idx)
        else:
            xshift = self.xshift[sal_idx]
            yshift = self.yshift[sal_idx]
            rot_theta = self.rot_theta[sal_idx]
            xscale = self.xscale[sal_idx]
            yscale = self.yscale[sal_idx]
            xshear = self.xshear[sal_idx]
            yshear = self.yshear[sal_idx]

        T = torch.cat([ # Translation Matrix
            torch.stack([torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev), xshift]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev), yshift]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        R = torch.cat([ # Rotation Matrix
            torch.stack([torch.cos(rot_theta), -torch.sin(rot_theta), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.sin(rot_theta), torch.cos(rot_theta), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        Sc = torch.cat([ # Scaling Matrix
            torch.stack([xscale, torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), yscale, torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        Sh = torch.cat([ # Shear Matrix
            torch.stack([torch.tensor(1.0).to(dev), xshear, torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([yshear, torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)

        theta = T @ R @ Sc @ Sh

        if inverse:
            # Invert the transformation matrix
            theta = torch.linalg.inv(theta)
        theta = theta[0:2,:] # slice into submatrix expected by affine_grid
        return theta

    def forward(self, x, sal_idx=0, inverse=False):
        '''Regrids input image based on affine transformation parameters.'''
        dev = x.device # assuming x and model are on the same device
        N, C, H, W = x.shape
        # # Apply soft constraints to parameters
        # if self.boundaries and self.constrain_params:
        #     xshift, yshift, rot_theta, xscale, yscale, xshear, yshear = self.get_constrained_params(sal_idx=sal_idx)
        # else:
        #     xshift = self.xshift[sal_idx]
        #     yshift = self.yshift[sal_idx]
        #     rot_theta = self.rot_theta[sal_idx]
        #     xscale = self.xscale[sal_idx]
        #     yscale = self.yscale[sal_idx]
        #     xshear = self.xshear[sal_idx]
        #     yshear = self.yshear[sal_idx]

        # T = torch.cat([ # Translation Matrix
        #     torch.stack([torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev), xshift.to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev), yshift.to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        # ], dim=0)
        # R = torch.cat([ # Rotation Matrix
        #     torch.stack([torch.cos(rot_theta.to(dev)), -torch.sin(rot_theta.to(dev)), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([torch.sin(rot_theta.to(dev)), torch.cos(rot_theta.to(dev)), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        # ], dim=0)
        # Sc = torch.cat([ # Scaling Matrix
        #     torch.stack([xscale.to(dev), torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), yscale.to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        # ], dim=0)
        # Sh = torch.cat([ # Shear Matrix
        #     torch.stack([torch.tensor(1.0).to(dev), xshear.to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([yshear.to(dev), torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
        #     torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        # ], dim=0)

        # theta = T @ R @ Sc @ Sh

        # if inverse:
        #     # Invert the transformation matrix
        #     theta = torch.linalg.inv(theta)

        # theta = theta[0:2,:] # slice into submatrix expected by affine_grid
        theta = self.get_affine_transform(sal_idx=sal_idx, inverse=inverse).to(dev)
        theta = theta.repeat(N,1,1)
        grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=True)
        if self.circular:
            grid = wrap_grid_horizontally(grid) # wrap x coordinates if electrodes around arm
        xresamp = torch.nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=True)
        return xresamp
    
    def reset_params(self, Tx=torch.tensor([0.0,0.0]), Ty=torch.tensor([0.0,0.0]), rot_theta=torch.tensor([0.0,0.0]), xscale=torch.tensor([1.0,1.0]), yscale=torch.tensor([1.0,1.0]), xshear=torch.tensor([0.0,0.0]), yshear=torch.tensor([0.0,0.0])):
        """Manually updates SAL parameters given a new set of parameters"""
        with torch.no_grad():
            for sal_idx in range(self.nsals):
                self.xshift[sal_idx].copy_(Tx[sal_idx])
                self.yshift[sal_idx].copy_(Ty[sal_idx])
                self.rot_theta[sal_idx].copy_(rot_theta[sal_idx])
                self.xscale[sal_idx].copy_(xscale[sal_idx])
                self.yscale[sal_idx].copy_(yscale[sal_idx])
                self.xshear[sal_idx].copy_(xshear[sal_idx])
                self.yshear[sal_idx].copy_(yshear[sal_idx])

        
    # def restart(self):
    #     '''Reinitialize the parameters of the affine transformation.'''
    #     bnds = [2*2.5/(self.input_shape[1]-1), 2*2.5/(self.input_shape[0]-1), 15/180, 0.1, 0.1, 0.1, 0.1]
    #     for sal_idx in range(len(self.xshift)):
    #         self.xshift[sal_idx].data = torch.tensor(random.uniform(-bnds[0], bnds[0])).to(self.xshift.device)
    #         self.yshift[sal_idx].data = torch.tensor(random.uniform(-bnds[1], bnds[1])).to(self.yshift.device)
    #         self.rot_theta[sal_idx].data = torch.tensor(random.uniform(-bnds[2], bnds[2])).to(self.rot_theta.device)
    #         self.xscale[sal_idx].data = torch.tensor(random.uniform(1-bnds[3], 1+bnds[3])).to(self.xscale.device)
    #         self.yscale[sal_idx].data = torch.tensor(random.uniform(1-bnds[4], 1+bnds[4])).to(self.yscale.device)
    #         self.xshear[sal_idx].data = torch.tensor(random.uniform(-bnds[5], bnds[5])).to(self.xshear.device)
    #         self.yshear[sal_idx].data = torch.tensor(random.uniform(-bnds[6], bnds[6])).to(self.yshear.device)   
        

class SpatialAdaptationHyser(SpatialAdaptation):
    def __init__(self, *args, T = True, R = True, Sc = True, Sh = True, **kwargs):
        super().__init__(*args, T=T, R=R, Sc=Sc, Sh=Sh, **kwargs)
        self.nsals = 2
        self.H = self.input_shape[0] // 2
        self.W = self.input_shape[1]
        self.xshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(self.nsals)]) 
        self.yshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(self.nsals)])
        self.rot_theta = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=R) for idx in range(self.nsals)]) # theta in rads.
        self.xscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(self.nsals)])
        self.yscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(self.nsals)])
        self.xshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(self.nsals)])
        self.yshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(self.nsals)])
    
    def forward(self, x, inverse=False):
        xtop = x[:, :, :self.H, :] # top half
        xbot = x[:, :, self.H:, :] # bottom half
        xtop = super().forward(xtop, sal_idx=0, inverse=inverse) # perform image resampling step
        xbot = super().forward(xbot, sal_idx=1, inverse=inverse)
        x = torch.cat((xtop, xbot), dim=2) # concatenate the two halves   
        return x


# # Inherits from Hyser module to have double SAL layer
# class SpatialAdaptationGrabmyo(SpatialAdaptation):
#     def __init__(self, *args, T = True, R = True, Sc = True, Sh = True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.H = [2, 2]
#         self.W = [8, 6]
#         self.xshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(2)]) 
#         self.yshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(2)])
#         self.rot_theta = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=R) for idx in range(2)]) # theta in rads.
#         self.xscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(2)])
#         self.yscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(2)])
#         self.xshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(2)])
#         self.yshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(2)])

#     def forward(self, x, inverse=True):
#         xforearm = x[:, :, :, :self.W[0]] # top half
#         xwrist = x[:, :, :, self.W[0]:] # bottom half
#         xforearm = super().forward(xforearm, sal_idx=0) # perform image resampling step
#         xwrist = super().forward(xwrist, sal_idx=1) # perform image resampling step
#         x = torch.cat((xforearm, xwrist), dim=3) # concatenate the two halves   
#         return x
    

## Inherits from Hyser module to have double SAL layer
# class SpatialAdaptationGrabmyo(SpatialAdaptation):
#     def __init__(self, *args, T = True, R = True, Sc = True, Sh = True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.H = [2, 2]
#         self.W = [8, 6]
#         self.xshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(2)]) 
#         self.yshift = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) for idx in range(2)])
#         self.rot_theta = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=R) for idx in range(2)]) # theta in rads.
#         self.xscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(2)])
#         self.yscale = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc) for idx in range(2)])
#         self.xshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(2)])
#         self.yshear = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh) for idx in range(2)])

#     def forward(self, x, inverse=False):
#         xforearm = x[:, :, :, :self.W[0]] # top half
#         xwrist = x[:, :, :, self.W[0]:] # bottom half
#         xforearm = super().forward(xforearm, sal_idx=0) # perform image resampling step
#         xwrist = super().forward(xwrist, sal_idx=1) # perform image resampling step
#         x = torch.cat((xforearm, xwrist), dim=3) # concatenate the two halves   
#         return x
    
    
## Median pooling utils and function
def unpack_param_2d(param):

  try:
    p_H, p_W = param[0], param[1]
  except:
    p_H, p_W = param, param

  return p_H, p_W

# def median_pool_2d(input, kernel_size=3, stride=1, padding=1, dilation=1):

#   #Input should be 4D (BCHW)
#   assert(input.dim() == 4)

#   #Get input dimensions
#   b_size, c_size, h_size, w_size = input.size()

#   #Get input parameters
#   k_H, k_W = unpack_param_2d(kernel_size)
#   s_H, s_W = unpack_param_2d(     stride)
#   p_H, p_W = unpack_param_2d(    padding)
#   d_H, d_W = unpack_param_2d(   dilation)

#   #First we unfold all the (kernel_size x kernel_size)  patches
#   unf_input =torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)

#   #Reshape it so that each patch is a column
#   row_unf_input = unf_input.reshape(b_size, c_size, k_H*k_W, -1)

#   #Apply median operation along the columns for each channel separately 
#   med_unf_input, med_unf_indexes =torch.median(row_unf_input, dim = 2, keepdim=True)

#   #Restore original shape
#   out_W = math.floor(((w_size + (2 * p_W) - (d_W * (k_W - 1)) - 1) / s_W) + 1)
#   out_H = math.floor(((h_size + (2 * p_H) - (d_H * (k_H - 1)) - 1) / s_H) + 1)

#   return med_unf_input.reshape(b_size, c_size, out_H, out_W)

# def median_pool_2d(input, kernel_size=3, stride=1, padding=1, dilation=1, circular=False):
#     """
#     Median pooling with optional circular padding along the horizontal (width) dimension.
#     Input should be 4D (BCHW).
#     """
#     assert(input.dim() == 4)

#     # Ensure all arguments are tuples
#     if isinstance(kernel_size, int):
#         kernel_size = (kernel_size, kernel_size)
#     if isinstance(stride, int):
#         stride = (stride, stride)
#     if isinstance(padding, int):
#         padding = (padding, padding)
#     if isinstance(dilation, int):
#         dilation = (dilation, dilation)

#     b_size, c_size, h_size, w_size = input.size()
#     k_H, k_W = kernel_size
#     s_H, s_W = stride
#     p_H, p_W = padding
#     d_H, d_W = dilation

#     # Circular pad along width if requested
#     if circular:
#         pad_left = (k_W - 1) // 2
#         pad_right = k_W // 2
#         input = torch.cat([input[..., -pad_left:], input, input[..., :pad_right]], dim=-1)
#         p_W = 0  # No zero-padding
#         out_W = w_size  # Output width remains the same as input width
#     else:
#         out_W = math.floor(((w_size + (2 * p_W) - (d_W * (k_W - 1)) - 1) / s_W) + 1)
#     out_H = math.floor(((h_size + (2 * p_H) - (d_H * (k_H - 1)) - 1) / s_H) + 1)

#     unf_input = torch.nn.functional.unfold(input, (k_H, k_W), dilation, (p_H, p_W), stride)
#     row_unf_input = unf_input.reshape(b_size, c_size, k_H * k_W, -1)
#     med_unf_input, _ = torch.median(row_unf_input, dim=2, keepdim=True)
#     med_unf_input = med_unf_input.reshape(b_size, c_size, out_H, out_W)

#     return med_unf_input

def median_pool_2d(input, kernel_size=3, stride=1, padding=1, dilation=1, circular=False):
    """
    Median pooling with optional circular padding along the horizontal (width) dimension,
    and always reflect padding along the vertical (height) dimension.
    Input should be 4D (BCHW).
    """
    assert(input.dim() == 4)

    # Ensure all arguments are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    b_size, c_size, h_size, w_size = input.size()
    k_H, k_W = kernel_size
    s_H, s_W = stride
    p_H, p_W = padding
    d_H, d_W = dilation

    if circular:
        # Circular pad along width
        pad_left = (k_W - 1) // 2
        pad_right = k_W // 2
        input = torch.cat([input[..., -pad_left:], input, input[..., :pad_right]], dim=-1)
        # Reflect pad along height
        if p_H > 0:
            input = torch.nn.functional.pad(input, (0, 0, p_H, p_H), mode='reflect')
        unfold_padding = (0, 0)  # Already padded
        out_W = w_size  # Output width remains the same as input width
        out_H = math.floor(((h_size + 2 * p_H - d_H * (k_H - 1) - 1) / s_H) + 1)
    else:
        # Reflect pad along both height and width
        if p_H > 0 or p_W > 0:
            input = torch.nn.functional.pad(input, (p_W, p_W, p_H, p_H), mode='reflect')
        unfold_padding = (0, 0)
        out_W = math.floor(((w_size + 2 * p_W - d_W * (k_W - 1) - 1) / s_W) + 1)
        out_H = math.floor(((h_size + 2 * p_H - d_H * (k_H - 1) - 1) / s_H) + 1)

    unf_input = torch.nn.functional.unfold(input, (k_H, k_W), dilation=dilation, padding=unfold_padding, stride=stride)
    row_unf_input = unf_input.reshape(b_size, c_size, k_H * k_W, -1)
    med_unf_input, _ = torch.median(row_unf_input, dim=2, keepdim=True)
    med_unf_input = med_unf_input.reshape(b_size, c_size, out_H, out_W)

    return med_unf_input


## Locally connected module needed for CapgmyoNet 
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class FactorizedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(FactorizedDepthwiseSeparableConv, self).__init__()

        # Factorize the kernel size (3, 3) into (3, 1) and (1, 3) or custom if provided
        k1, k2 = kernel_size

        # Depthwise k1x1 Convolution
        self.depthwise_k1 = nn.Conv2d(in_channels, in_channels, kernel_size=(k1, 1), 
                                      stride=stride, padding=(padding, 0), groups=in_channels)

        # Depthwise 1xk2 Convolution
        self.depthwise_k2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k2), 
                                      stride=stride, padding=(0, padding), groups=in_channels)

        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise_k1(x)
        x = self.depthwise_k2(x)
        x = self.pointwise(x)
        return x