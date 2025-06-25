# This file will contain all the different classification networks that we will be running
# Potential classifier ideas are: multiclass SVM, LDA, MLP, TCN, Multiclass ADaboost, Bayesian Networks.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks_utils import SpatialAdaptation, SpatialAdaptationHyser, LocallyConnected2d
from torchvision import models

# Canonical EMG network from original capgmyo paper
class CapgMyoNet(nn.Module):
    def __init__(self, num_classes=8, input_shape=(8, 16), channels=64, kernel_sz=3, baseline=True, input_transform_name='spatial-adaptation', p_input=0.0, track_running_stats=True, circular=False, boundaries=None):
        super(CapgMyoNet, self).__init__()

        self.channels = channels
        self.kernel_sz = kernel_sz

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.corrective_gain = False

        if baseline:
            self.baseline = torch.nn.parameter.Parameter(torch.zeros(1, 1, input_shape[0], input_shape[1]))
        else:
            self.register_buffer('baseline', torch.zeros(1, 1, input_shape[0], input_shape[1])) # original coordinates

        self.input_dropout = nn.Dropout(p=p_input)

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

        # Set input transformation method
        self.adaptation_phase = False
        self.input_transform_name = input_transform_name
        if input_transform_name == 'spatial-adaptation':
            self.input_transform = SpatialAdaptation(input_shape, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'spatial-adaptation-hyser':
            self.input_transform = SpatialAdaptationHyser(input_shape, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'linear-layer':
            self.input_transform =  nn.Sequential(
                nn.Flatten(start_dim=1),  # Flatten from (B, 1, H, W) → (B, H*W)
                nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*input_shape[1]), # learnable input linear transformation            
                nn.Unflatten(dim=1, unflattened_size=(1, input_shape[0], input_shape[1]))  # Back to (B, 1, H, W)
            )
        else:
            self.input_transform = lambda x: x  # No transformation

    def report_features(self):
        return [self.channels, self.kernel_sz]

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_session_means(self, X1, X2):
        """
        Calculate the mean of the original input data.
        This is used to set the baseline for normalization.
        """
        # Assuming X is a tensor of shape (batch_size, channels, height, width)
        self.corrective_gain = True
        with torch.no_grad():
            self.register_buffer('mean_session1', X1.to(next(self.parameters()).device).mean(dim=0, keepdim=True))  # Store the mean as baseline
            self.register_buffer('mean_session2', X2.to(next(self.parameters()).device).mean(dim=0, keepdim=True))  # Store the mean as baseline

    def forward(self, x):
        if self.adaptation_phase and self.corrective_gain:
            mean_session2 = self.input_transform(self.mean_session2) # apply input transform to session 2 mean
            scaling_factors = (self.mean_session1 / (mean_session2 + 1e-12))
            scaling_factors = torch.clamp(scaling_factors, min=0.5, max=2.0)  # clamp scaling factors to avoid extreme values
            x = x * scaling_factors  # scale to match magnitude of session 1

        x = self.batchnorm0(x)
        if self.adaptation_phase:
            x = x - self.baseline # perform baseline normalization
            x = self.input_transform(x) # perform image resampling step
        x = self.input_dropout(x)
        # x = self.batchnorm0(x)
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
        return x.reshape(x.shape[0], self.num_classes)
    

class LogisticRegressor(nn.Module):

    def __init__(self, num_classes=8, input_shape=(8, 16), baseline=True, p_input=0.0, input_transform_name='spatial-adaptation', track_running_stats=True, circular=False, boundaries=None):
        super(LogisticRegressor, self).__init__()

        self.channels = input_shape[0] * input_shape[1]  # Flattened input shape

        self.input_shape = input_shape
        self.num_classes = num_classes

        # scaling = True
        self.corrective_gain = False
        
        if baseline:
            self.baseline = torch.nn.parameter.Parameter(torch.zeros(1, 1, input_shape[0], input_shape[1]))
        else:
            self.register_buffer('baseline', torch.zeros(1, 1, input_shape[0], input_shape[1])) # original coordinates

        self.input_dropout = nn.Dropout(p=p_input)
        self.bn = nn.BatchNorm2d(1, track_running_stats=track_running_stats)
        self.pca = False
        self.fc = nn.Linear(self.channels, self.num_classes)

        # Set input transformation method
        self.adaptation_phase = False
        self.input_transform_name = input_transform_name
        if input_transform_name == 'spatial-adaptation':
            self.input_transform = SpatialAdaptation(input_shape, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'spatial-adaptation-hyser':
            self.input_transform = SpatialAdaptationHyser(input_shape, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'linear-layer':
            self.input_transform =  nn.Sequential(
                nn.Flatten(start_dim=1),  # Flatten from (B, 1, H, W) → (B, H*W)
                nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*input_shape[1]), # learnable input linear transformation            
                nn.Unflatten(dim=1, unflattened_size=(1, input_shape[0], input_shape[1]))  # Back to (B, 1, H, W)
            )
        else:
            self.input_transform = lambda x: x  # No transformation        
        
    def get_session_means(self, X1, X2):
        """
        Calculate the mean of the original input data.
        This is used to set the baseline for normalization.
        """
        # Assuming X is a tensor of shape (batch_size, channels, height, width)
        self.corrective_gain = True
        with torch.no_grad():
            self.register_buffer('mean_session1', X1.to(next(self.parameters()).device).mean(dim=0, keepdim=True))  # Store the mean as baseline
            self.register_buffer('mean_session2', X2.to(next(self.parameters()).device).mean(dim=0, keepdim=True))  # Store the mean as baseline

    def forward(self, x):
        if self.adaptation_phase and self.corrective_gain:
            mean_session2 = self.input_transform(self.mean_session2) # apply input transform to session 2 mean
            scaling_factors = (self.mean_session1 / (mean_session2 + 1e-12))
            scaling_factors = torch.clamp(scaling_factors, min=0.5, max=2.0)  # clamp scaling factors to avoid extreme values
            x = x * scaling_factors  # scale to match magnitude of session 1
        
        x = self.bn(x) # applies normalization procedure after usual filtering operations
        if self.adaptation_phase:
            x = x - self.baseline # subtract baseline for baseline normalization
            x = self.input_transform(x) # perform image resampling step

        x = self.input_dropout(x)
        x = x.reshape(x.shape[0],-1) # flatten for determining classification
        if self.pca:
            x = torch.mm(x, self.pca_projection)  # apply PCA projection if available
        x = self.fc(x)
        return x.reshape(x.shape[0], self.num_classes)

class VGG11Net(nn.Module):
    def __init__(self, input_size=(8, 16), num_classes=8, baseline=True, p_input=0.0, input_transform_name='spatial-adaptation', track_running_stats=True, circular=False, boundaries=None):
        super(VGG11Net, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.corrective_gain = False

        # Baseline parameter/buffer
        if baseline:
            self.baseline = torch.nn.parameter.Parameter(torch.zeros(1, 1, input_size[0], input_size[1]))
        else:
            self.register_buffer('baseline', torch.zeros(1, 1, input_size[0], input_size[1]))

        self.input_dropout = nn.Dropout(p=p_input)
        self.batchnorm0 = nn.BatchNorm2d(1, track_running_stats=track_running_stats)

        # Input transformation
        self.adaptation_phase = False
        self.input_transform_name = input_transform_name
        if input_transform_name == 'spatial-adaptation':
            self.input_transform = SpatialAdaptation(input_size, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'spatial-adaptation-hyser':
            self.input_transform = SpatialAdaptationHyser(input_size, circular=circular, boundaries=boundaries)
        elif input_transform_name == 'linear-layer':
            self.input_transform = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_size[0]*input_size[1], input_size[0]*input_size[1]),
                nn.Unflatten(dim=1, unflattened_size=(1, input_size[0], input_size[1]))
            )
        else:
            self.input_transform = lambda x: x

        # VGG11 backbone
        self.vgg = models.vgg11(pretrained=False)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Calculate flattened size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])
            conv_output = self.vgg.features(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)

        self.vgg.classifier[0] = nn.Linear(flattened_size, 4096)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def get_session_means(self, X1, X2):
        self.corrective_gain = True
        with torch.no_grad():
            self.register_buffer('mean_session1', X1.to(next(self.parameters()).device).mean(dim=0, keepdim=True))
            self.register_buffer('mean_session2', X2.to(next(self.parameters()).device).mean(dim=0, keepdim=True))

    def forward(self, x):
        if self.adaptation_phase and self.corrective_gain:
            mean_session2 = self.input_transform(self.mean_session2)
            scaling_factors = (self.mean_session1 / (mean_session2 + 1e-12))
            scaling_factors = torch.clamp(scaling_factors, min=0.5, max=2.0)
            x = x * scaling_factors

        x = self.batchnorm0(x)
        if self.adaptation_phase:
            x = x - self.baseline
            x = self.input_transform(x)
        x = self.input_dropout(x)
        x = self.vgg(x)
        return x.reshape(x.shape[0], self.num_classes)