import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csc_array
from scipy.sparse.linalg import bicgstab
import scipy.sparse as sp


A = csc_array((data, (row, col)), shape=(n, n))

# Step 2: Prepare the identity matrix (for inversion)
I = np.eye(n)

# Step 3: Solve A @ X = I using bicgstab
A_inv = np.zeros((n, n))

for i in range(n):
    # Each column of I is a unit vector (e_i)
    b = I[:, i]
    
    # Solve for the i-th column of the inverse
    x, exit_code = bicgstab(A, b, tol=1e-6)
    if exit_code != 0:
        print(f"Warning: Solver did not converge for column {i}")
    
    A_inv[:, i] = x

# Step 4: Convert to PyTorch tensor if needed
A_inv_torch = torch.tensor(A_inv, dtype=torch.float32)
print(A_inv_torch.shape)

def get_grid(grid_shape, xshift, yshift, rot_theta, device='cpu'):
    """
    Compute the affine transformation matrix for translation and rotation.
    
    Args:
    - translation: Tuple (tx, ty) for horizontal and vertical translation.
    - rotation: Rotation angle in radians (counterclockwise).
    - device: Device to place the tensor on ('cpu' or 'cuda').
    
    Returns:
    - affine_matrix: A 2x3 affine transformation matrix.
    """
    H, W = grid_shape
    xshift, yshift, rot_theta = torch.tensor(xshift), torch.tensor(yshift), torch.tensor(rot_theta)
    T = torch.cat([ # Translation Matrix
        torch.stack([torch.tensor(1.0).to(device), torch.tensor(0.0).to(device), xshift]).unsqueeze(0),
        torch.stack([torch.tensor(0.0).to(device), torch.tensor(1.0).to(device), yshift]).unsqueeze(0),
        torch.stack([torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)]).unsqueeze(0)
    ], dim=0)
    R = torch.cat([ # Rotation Matrix
        torch.stack([torch.cos(rot_theta), -torch.sin(rot_theta), torch.tensor(0.0).to(device)]).unsqueeze(0),
        torch.stack([torch.sin(rot_theta), torch.cos(rot_theta), torch.tensor(0.0).to(device)]).unsqueeze(0),
        torch.stack([torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)]).unsqueeze(0)
    ], dim=0)

    theta = R @ T
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(1,1,1)
    grid = torch.nn.functional.affine_grid(theta, size = (1,1,H, W), align_corners=True)
    return grid

def generate_bilinear_transformation_matrix(grid, device='cpu'):
    """
    Generate the linear transformation matrix A for bilinear interpolation 
    that maps the image pixels to the transformed pixels.
    
    Args:
    - affine_matrix: The 2x3 affine transformation matrix.
    - H: Height of the image.
    - W: Width of the image.
    - device: Device to place the tensor on ('cpu' or 'cuda').
    
    Returns:
    - A: The dense matrix representing the linear transformation (size H*W x H*W).
    """    
    # Flatten grid to (2, H * W), where the first row is x-coordinates and second row is y-coordinates
    H, W = grid.shape[1:3]
    grid_flat = grid.view(-1, 2).T  # Shape: (2, H * W)
    
    grid_x = (grid[..., 0] + 1) * (W - 1) / 2  # Mapping x-coordinates to [0, W-1]
    grid_y = (grid[..., 1] + 1) * (H - 1) / 2  # Mapping y-coordinates to [0, H-1]

    # Flatten grid to (2, H * W), where the first row is x-coordinates and second row is y-coordinates
    grid_flat = torch.stack([grid_x.view(-1), grid_y.view(-1)], dim=0)  # Shape: (2, H * W)
    
    # Step 2: Create grid indices
    x = grid_flat[0, :]  # x-coordinates
    y = grid_flat[1, :]  # y-coordinates

    # Calculate the integer and fractional parts of x and y
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Get the fractional parts for interpolation
    dx = x - x0.float()
    dy = y - y0.float()

    # Step 3: Generate the dense transformation matrix A (H * W x H * W)
    A = torch.zeros(H * W, H * W, dtype=torch.float32, device=device)  # Dense matrix
    
    # Compute the bilinear weights for each pixel without using loops
    # Bilinear interpolation formula for weights at (x0, y0), (x1, y0), (x0, y1), (x1, y1)
    weight00 = (1 - dx) * (1 - dy)
    weight01 = (1 - dx) * dy
    weight10 = dx * (1 - dy)
    weight11 = dx * dy

    # Compute the linear indices for all four surrounding pixels
    idx00 = y0 * W + x0
    idx01 = y0 * W + x1
    idx10 = y1 * W + x0
    idx11 = y1 * W + x1
    
    # Valid index masks (inside the image bounds)
    valid_x0 = (x0 >= 0) & (x0 < W)
    valid_x1 = (x1 >= 0) & (x1 < W)
    valid_y0 = (y0 >= 0) & (y0 < H)
    valid_y1 = (y1 >= 0) & (y1 < H)

    valid00 = valid_x0 & valid_y0
    valid01 = valid_x1 & valid_y0
    valid10 = valid_x0 & valid_y1
    valid11 = valid_x1 & valid_y1

    # Apply valid masks to indices and weights
    A[torch.arange(H * W)[valid00], idx00[valid00]] = weight00[valid00]
    A[torch.arange(H * W)[valid01], idx01[valid01]] = weight01[valid01]
    A[torch.arange(H * W)[valid10], idx10[valid10]] = weight10[valid10]
    A[torch.arange(H * W)[valid11], idx11[valid11]] = weight11[valid11]

    return A

if __name__ == '__main__':
    H, W = 20, 20
    a = torch.zeros(H, W)
    a[8:H-8, 8:W-8] = 1.0
    plt.figure()
    sns.heatmap(a)
    plt.savefig('square')

    grid = get_grid(a.shape, 2*9/W, 0.0, 0.0)
    A = generate_bilinear_transformation_matrix(grid)
    flat_a = a.view(-1)
    flat_a_transformed = A @ flat_a
    a_transformed = flat_a_transformed.view(a.shape)
    plt.figure()
    sns.heatmap(a_transformed)
    plt.savefig('transformed_square')
    print()