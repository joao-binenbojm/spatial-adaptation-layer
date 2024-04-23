# This file will contain all the different classification networks that we will be running
# Potential classifier ideas are: multiclass SVM, LDA, MLP, TCN, Multiclass ADaboost, Bayesian Networks.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math

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


class CapgMyoNet(nn.Module):
    def __init__(self, num_classes=8, input_shape=(8, 16), channels=64, kernel_sz=3, track_running_stats=True):
        super(CapgMyoNet, self).__init__()

        self.channels = channels
        self.kernel_sz = kernel_sz

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.batchnorm0 = nn.BatchNorm2d(1, track_running_stats=track_running_stats)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(kernel_sz, kernel_sz), stride=(1, 1), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(kernel_sz, kernel_sz), stride=(1, 1), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu2 = nn.ReLU()

        self.localconv3 = LocallyConnected2d(channels, channels, kernel_size=1, stride=(1, 1), output_size=input_shape)
        self.batchnorm3 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu3 = nn.ReLU()

        self.localconv4 = LocallyConnected2d(channels, channels, kernel_size=1, stride=(1, 1), output_size=input_shape)
        self.batchnorm4 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(input_shape[0] * input_shape[1] * channels, 512)
        self.batchnorm5 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(512, 512)
        self.batchnorm6 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(512, 128)
        self.batchnorm7 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(128, self.num_classes)
        self.sm = nn.Softmax(dim=2)

        # self.apply(CapMyoNet.init_weights)

    def report_features(self):
        return [self.channels, self.kernel_sz]

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):

        # # Reshape EMG channels into grid
        # batch_size = x.shape[0]
        # x = x.squeeze() # remove any redundant dimensions
        # x = x.view(batch_size, self.input_shape[1], self.input_shape[0]) # convert from channels into grid
        # x = torch.transpose(x, 1, 2) # tranpose required as reshape is not the default ordering

        x = self.batchnorm0(x)
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.relu2(self.batchnorm2(self.conv2(x)))

        x = self.relu3(self.batchnorm3(self.localconv3(x)))
        x = self.dropout4(self.relu4(self.batchnorm4(self.localconv4(x))))

        x = x.reshape(x.shape[0], 1, -1)
        x = self.fc5(x)
        x = self.dropout5(self.relu5(self.batchnorm5(x)))
        x = self.dropout6(self.relu6(self.batchnorm6(self.fc6(x))))
        x = self.relu7(self.batchnorm7(self.fc7(x)))
        x = self.fc8(x)
        # x = self.sm(x)
        return x.reshape(x.shape[0], self.num_classes)
    


class Shift(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.Nv, self.Nh = input_shape
        self.xshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.yshift = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.register_buffer('yreg', torch.arange(self.Nv)*10) # original coordinates
        self.register_buffer('xreg', torch.arange(self.Nh)*10) # original coordinates

    def forward(self, x):
        '''Regrids input image based on shift parameters.'''
        yreg = self.yreg - self.yshift*10
        xreg = self.xreg - self.xshift*10
        H, W = self.Nv*10, self.Nh*10
        xreg, yreg = 2*xreg/(W) - 1, 2*yreg/(H) - 1 # scale between -1 and 1
        grid_y, grid_x = torch.meshgrid(yreg, xreg, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, self.Nv, self.Nh, 2)
        grid = grid.repeat(x.shape[0], 1, 1, 1) # get grid to match batch size dimensions
        xresamp = torch.nn.functional.grid_sample(x, grid)
        return xresamp


class RMSNet(nn.Module):

    def __init__(self, num_classes=8, input_shape=(8, 16), channels=64, kernel_sz=3, baseline=True, track_running_stats=True):
        super(RMSNet, self).__init__()

        self.channels = channels
        self.kernel_sz = kernel_sz

        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # if baseline:
            # self.baseline = torch.nn.parameter.Parameter(torch.zeros(1, 1, input_shape[0], input_shape[1]))
        # else:
            # self.register_buffer('baseline', torch.zeros(1, 1, input_shape[0], input_shape[1])) # original coordinates

        self.bn = nn.BatchNorm2d(1, track_running_stats=track_running_stats)
        self.shift = Shift(input_shape)
        self.fc = nn.Linear(self.channels, self.num_classes)
        # self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        x = median_pool_2d(x) # perform median filtering step
        x = self.bn(x) # applies normalization procedure after usual filtering operations
        # x = x - self.baseline # subtract baseline for baseline normalization
        x = self.shift(x) # perform image resampling step
        x = x.view(x.shape[0],-1) # flatten for determining classification
        x = self.fc(x)
        # x = self.sm(x)
        return x.reshape(x.shape[0], self.num_classes)

    
class CapgMyoNetInterpolate(nn.Module):
    def __init__(self, num_classes=8, input_shape=(8, 16), channels=64, kernel_sz=3, baseline=True, track_running_stats=True):
        super(CapgMyoNetInterpolate, self).__init__()

        self.channels = channels
        self.kernel_sz = kernel_sz

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.shift = Shift(input_shape)

        if baseline:
            self.baseline = torch.nn.parameter.Parameter(torch.zeros(1, 1, input_shape[0], input_shape[1]))
        else:
            self.register_buffer('baseline', torch.zeros(1, 1, input_shape[0], input_shape[1])) # original coordinates


        self.batchnorm0 = nn.BatchNorm2d(1, track_running_stats=track_running_stats)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(kernel_sz, kernel_sz), stride=(1, 1), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(kernel_sz, kernel_sz), stride=(1, 1), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu2 = nn.ReLU()

        self.localconv3 = LocallyConnected2d(channels, channels, kernel_size=1, stride=(1, 1), output_size=input_shape)
        self.batchnorm3 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu3 = nn.ReLU()

        self.localconv4 = LocallyConnected2d(channels, channels, kernel_size=1, stride=(1, 1), output_size=input_shape)
        self.batchnorm4 = nn.BatchNorm2d(channels, track_running_stats=track_running_stats)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(input_shape[0] * input_shape[1] * channels, 512)
        self.batchnorm5 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(512, 512)
        self.batchnorm6 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(512, 128)
        self.batchnorm7 = nn.BatchNorm1d(1, track_running_stats=track_running_stats)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(128, self.num_classes)
        self.sm = nn.Softmax(dim=2)

        # self.apply(CapMyoNet.init_weights)

    def report_features(self):
        return [self.channels, self.kernel_sz]

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):

        # # Reshape EMG channels into grid
        # batch_size = x.shape[0]
        # x = x.squeeze() # remove any redundant dimensions
        # x = x.view(batch_size, self.input_shape[1], self.input_shape[0]) # convert from channels into grid
        # x = torch.transpose(x, 1, 2) # tranpose required as reshape is not the default ordering

        x = self.shift(x) # perform image resampling step
        x = x - self.baseline # perform baseline normalization
        x = self.batchnorm0(x)
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.relu2(self.batchnorm2(self.conv2(x)))

        x = self.relu3(self.batchnorm3(self.localconv3(x)))
        x = self.dropout4(self.relu4(self.batchnorm4(self.localconv4(x))))

        x = x.reshape(x.shape[0], 1, -1)
        x = self.fc5(x)
        x = self.dropout5(self.relu5(self.batchnorm5(x)))
        x = self.dropout6(self.relu6(self.batchnorm6(self.fc6(x))))
        x = self.relu7(self.batchnorm7(self.fc7(x)))
        x = self.fc8(x)
        # x = self.sm(x)
        return x.reshape(x.shape[0], self.num_classes)


