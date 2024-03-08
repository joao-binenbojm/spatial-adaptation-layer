# This file will contain all the different classification networks that we will be running
# Potential classifier ideas are: multiclass SVM, LDA, MLP, TCN, Multiclass ADaboost, Bayesian Networks.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np


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

        # Reshape EMG channels into grid
        batch_size = x.shape[0]
        x = x.squeeze() # remove any redundant dimensions
        x = x.view(batch_size, self.input_shape[1], self.input_shape[0]) # convert from channels into grid
        x = torch.transpose(x, 1, 2) # tranpose required as reshape is not the default ordering

        x = x[:, None, :, :]
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