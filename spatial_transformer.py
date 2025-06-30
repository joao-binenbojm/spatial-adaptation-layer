import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSTN(nn.Module):
    """
    Convolutional Spatial Transformer Network optimized for small grayscale images
    Input shapes: 8x16 or 7x24 (or similar small dimensions)
    """
    def __init__(self, input_shape=(32,32)):
        super(ConvSTN, self).__init__()
        max_height, max_width = input_shape
        input_channels = 1
        
        # Localization network - optimized for rectangular inputs
        # Handles shapes like 8x16, 7x24, 32x8 efficiently
        self.localization = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 8x16->4x8, 7x24->3x12, 32x8->16x4
            
            # Second conv block  
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 4x8->2x4, 3x12->1x6, 16x4->8x2
            
            # Third conv block - no pooling to preserve spatial info for small dimensions
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            # Fourth conv block for better feature extraction
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        
        # Adaptive pooling to handle rectangular shapes gracefully
        # Uses asymmetric pooling to preserve aspect ratio information
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))
        
        # Regression head optimized for rectangular inputs
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 2 * 4, 256),  # 2x4 = 8 spatial locations
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 6)  # 6 parameters for affine transformation
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Get input dimensions
        batch_size, channels, height, width = x.size()
        
        # Localization network forward pass
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(batch_size, -1)
        
        # Predict transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # Reshape to [batch_size, 2, 3] for affine_grid
        
        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # Sample from input using the grid
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        # print(theta)
        return x_transformed, theta