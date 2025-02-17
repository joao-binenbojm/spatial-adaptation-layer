import torch

def central_average(x):
    sorted_x, _ = torch.sort(x, dim=-1)  # Sort along the last dimension
    n = x.shape[-1]
    
    lower_idx = int(n * 0.25)  # 25th percentile index
    upper_idx = int(n * 0.75)  # 75th percentile index
    
    central_values = sorted_x[lower_idx:upper_idx]  # Extract central 50%
    return central_values.mean(dim=-1)  # Average the central values

class KurtosisLoss(torch.nn.Module):
    def __init__(self):
        super(KurtosisLoss, self).__init__()

    def forward(self, Y):
        # Y is assumed to have shape (batch_size, num_components)
        
        # Calculate the mean and variance of each component along the batch dimension
        mean_Y = Y.mean(dim=0, keepdim=True)
        centered_Y = (Y - mean_Y) / (Y.std(dim=0, keepdim=True) + 1e-8)

        # Calculate kurtosis for each component
        # Fourth moment: E[Y_i^4]
        fourth_moment = torch.mean(centered_Y ** 4, dim=0)
        
        # Second moment (variance): E[Y_i^2]
        second_moment = torch.mean(centered_Y ** 2, dim=0)
        
        # Kurtosis for each component: (E[Y_i^4] / (E[Y_i^2])^2) - 3
        kurtosis = fourth_moment / (second_moment ** 2 + 1e-8) - 3
        
        # Loss as negative absolute kurtosis to maximize independence
        # print(f'Kurtosis: max={kurtosis.max()}, min={kurtosis.min()}, mean={kurtosis.mean()}')
        # loss = -torch.mean(kurtosis)
        loss = -central_average(kurtosis)
        return loss

class NegentropyLoss(torch.nn.Module):
    def __init__(self):
        super(NegentropyLoss, self).__init__()

    def forward(self, y):
        # Enforce input y is zero-mean and unit variance
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-8)
        
        # Log-cosh contrast function
        G_y_logcosh = torch.log(torch.cosh(y))
        G_v_logcosh = torch.log(torch.cosh(torch.randn_like(y)))

        # Square contrast function
        # negentropy = torch.mean(torch.square(y)) - torch.log(torch.cosh(torch.randn_like(y)))
        
        # Exponential contrast function
        G_y_exponential = -torch.exp(-y**2 / 2)
        G_v_exponential = -torch.exp(-torch.randn_like(y)**2 / 2)
        
        # Combine both contrast functions for negentropy
        negentropy_logcosh = torch.mean(G_y_logcosh) - torch.mean(G_v_logcosh)
        negentropy_exponential = torch.mean(G_y_exponential) - torch.mean(G_v_exponential)
        
        # Sum both to form the final combined negentropy
        negentropy = negentropy_logcosh**2 + negentropy_exponential**2
        
        # Return the negative to make this a loss function (minimize -J(y))
        return -negentropy