import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math


class Shift(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.Nv, self.Nh = input_shape
        self.xshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.yshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.register_buffer('yreg', torch.arange(self.Nv)) # original coordinates
        self.register_buffer('xreg', torch.arange(self.Nh)) # original coordinates

    def forward(self, x):
        '''Regrids input image based on shift parameters.'''
        yreg = self.yreg - self.yshift
        xreg = self.xreg - self.xshift
        H, W = self.Nv, self.Nh
        xreg, yreg = 2*xreg/(W) - 1, 2*yreg/(H) - 1 # scale between -1 and 1
        grid_y, grid_x = torch.meshgrid(yreg, xreg, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, self.Nv, self.Nh, 2)
        grid = grid.repeat(x.shape[0], 1, 1, 1) # get grid to match batch size dimensions
        xresamp = torch.nn.functional.grid_sample(x, grid)
        return xresamp


class SpatialAdaptation(torch.nn.Module):
    def __init__(self, input_shape, T = True, R = True, Sc = True, Sh = True):
        super().__init__()
        self.Nv, self.Nh = input_shape
        self.register_buffer('yreg', torch.arange(self.Nv)) # original coordinates
        self.register_buffer('xreg', torch.arange(self.Nh)) # original coordinates
        # make the scaling...
        self.xshift = torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) 
        self.yshift = torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=T) 
        self.rot_theta = torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=R) # theta in rads.
        self.xscale = torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc)
        self.yscale = torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=Sc)
        self.xshear = torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh)
        self.yshear = torch.nn.parameter.Parameter(torch.tensor(0.0), requires_grad=Sh)
  
    def forward(self, x):
        '''Regrids input image based on affine shift parameters.'''
        dev = x.device # assuming x and model are on the same device
        N, C, _, _ = x.shape
        H, W = self.Nv, self.Nh
        # xshift_std, yshift_std = 2*self.xshift/(W), 2*self.yshift/(H) # scale shifts in the range (-1 to 1) to keep xshift and yshift in pixel coords
        xshift_std, yshift_std = self.xshift, self.yshift # scale shifts in the range (-1 to 1) to keep xshift and yshift in pixel coords
        T = torch.cat([ # Translation Matrix
            torch.stack([torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev), xshift_std]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev), yshift_std]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        R = torch.cat([ # Rotation Matrix
            torch.stack([torch.cos(self.rot_theta), -torch.sin(self.rot_theta), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.sin(self.rot_theta), torch.cos(self.rot_theta), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        Sc = torch.cat([ # Scaling Matrix
            torch.stack([self.xscale, torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), self.yscale, torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)
        Sh = torch.cat([ # Shear Matrix
            torch.stack([torch.tensor(1.0).to(dev), self.xshear, torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([self.yshear, torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev), torch.tensor(1.0).to(dev)]).unsqueeze(0)
        ], dim=0)

        theta = Sh @ Sc @ R @ T
        theta = theta[0:2,:] # slice into submatrix expected by affine_grid
        theta = theta.repeat(N,1,1)
        grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W))
        xresamp = torch.nn.functional.grid_sample(x, grid)
        
        return xresamp

## Median pooling utils and function
def unpack_param_2d(param):

  try:
    p_H, p_W = param[0], param[1]
  except:
    p_H, p_W = param, param

  return p_H, p_W

def median_pool_2d(input, kernel_size=3, stride=1, padding=1, dilation=1):

  #Input should be 4D (BCHW)
  assert(input.dim() == 4)

  #Get input dimensions
  b_size, c_size, h_size, w_size = input.size()

  #Get input parameters
  k_H, k_W = unpack_param_2d(kernel_size)
  s_H, s_W = unpack_param_2d(     stride)
  p_H, p_W = unpack_param_2d(    padding)
  d_H, d_W = unpack_param_2d(   dilation)

  #First we unfold all the (kernel_size x kernel_size)  patches
  unf_input =torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)

  #Reshape it so that each patch is a column
  row_unf_input = unf_input.reshape(b_size, c_size, k_H*k_W, -1)

  #Apply median operation along the columns for each channel separately 
  med_unf_input, med_unf_indexes =torch.median(row_unf_input, dim = 2, keepdim=True)

  #Restore original shape
  out_W = math.floor(((w_size + (2 * p_W) - (d_W * (k_W - 1)) - 1) / s_W) + 1)
  out_H = math.floor(((h_size + (2 * p_H) - (d_H * (k_H - 1)) - 1) / s_H) + 1)

  return med_unf_input.reshape(b_size, c_size, out_H, out_W)

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
