import numpy as np
import matplotlib.pyplot as plt
import torch


def apply_affine(emg_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1):
    '''Applies an affine transformation to grid coordinates prior to downsampling to simulate a near-perfect interpolation.'''

    N, C, H, W = emg_grid.shape
    Tx, Ty = torch.tensor(2*Tx/W), torch.tensor(2*Ty/H) # Normalize translation values automatically
    theta, xscale, yscale = torch.tensor(theta) / torch.pi, torch.tensor(xscale), torch.tensor(yscale)

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

    # theta = Sc @ R @ T # learning order
    theta = T @ R @ Sc
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(N,1,1)
    grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=False)
    xresamp = torch.nn.functional.grid_sample(emg_grid, grid)
    
    return xresamp


# Create an image with constant pixel values
image = torch.ones((1, 1, 100, 100)) * 128  # 100x100 image with pixel value 128 (gray)

# Rotate the image by 5 degrees
rotated_image = apply_affine(image, theta=20*np.pi/180)
image, rotated_image = image.squeeze(), rotated_image.squeeze()

# Display the original and rotated images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(rotated_image, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Rotated Image (5°)")
axes[1].axis("off")

plt.savefig("rot.png")
