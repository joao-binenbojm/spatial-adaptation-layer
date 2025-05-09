import torch
from networks_utils import wrap_grid_horizontally

def get_transformation(grid_shape, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, xshear=0, yshear=0, inverse=False):
    '''Computes the transformed grid coordinates for euclidina distance comparison.'''

    N, C, H, W = grid_shape
    Tx, Ty = torch.tensor(Tx), torch.tensor(Ty) # Normalize translation values automatically
    theta, xscale, yscale = torch.tensor(theta), torch.tensor(xscale), torch.tensor(yscale)
    xshear, yshear = torch.tensor(xshear), torch.tensor(yshear)

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
    Sh = torch.cat([ # Shear Matrix
            torch.stack([torch.tensor(1.0), xshear, torch.tensor(0.0)]).unsqueeze(0),
            torch.stack([yshear, torch.tensor(1.0), torch.tensor(0.0)]).unsqueeze(0),
            torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).unsqueeze(0)
        ], dim=0)

    # theta = Sc @ R @ T # learning order
    if inverse:
        theta = Sh @ Sc @ R @ T
    else:
        theta = T @ R @ Sc @ Sh
    return theta
    # theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    # theta = theta.repeat(N,1,1)
    # # Obtain transformed grid in pixel units
    # grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=False)
    # return grid


def apply_affine(data_grid, Tx=0, Ty=0, theta=0, xscale=1, yscale=1, xshear=0, yshear=0, mode='bilinear', circular=False):
    '''Apply affine transformation to a given input grid.'''
    theta = get_transformation(data_grid.shape, Tx=Tx, Ty=Ty, theta=theta, xscale=xscale, yscale=yscale, xshear=xshear, yshear=yshear)
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(data_grid.shape[0],1,1)
    grid = torch.nn.functional.affine_grid(theta, size=data_grid.shape, align_corners=False)
    if circular: # if cicular, wrap the grid horizontally
        grid = wrap_grid_horizontally(grid, data_grid.shape[2], data_grid.shape[3])
    data_resamp = torch.nn.functional.grid_sample(data_grid, grid, mode=mode, align_corners=False)
    return data_resamp

# def apply_affine(data_grid, theta, mode='bilinear', circular=False):
#     '''Apply affine transformation to a given input grid.'''
#     N, C, H, W = data_grid.shape
#     theta = theta[0:2,:] # slice into submatrix expected by affine_grid
#     theta = theta.repeat(N,1,1)
#     grid = torch.nn.functional.affine_grid(theta, size=data_grid.shape, align_corners=False)
#     if circular: # if cicular, wrap the grid horizontally
#         grid = wrap_grid_horizontally(grid, H, W)
#     data_resamp = torch.nn.functional.grid_sample(data_grid, grid, mode=mode)
#     return data_resamp

def grid_distance(grid1, grid2, IED=1):
    '''Computes Euclidian distance between grid coordinates of true and learned transformations. Returns distance in cm.'''
    dist = torch.linalg.norm((grid1*IED - grid2*IED), dim=3).mean() # compute average distance
    return dist.item()

def get_grid_distance(grid_shape, true_params, learned_params, IED=1, circular=False):
    '''Given the shape of a specific grid, the true and learned params, compute the average distance in cm of between corresponding electrodes of the original grid and final grid.'''
    N, _, H, W = grid_shape
    # Apply identity transformation to get original grid
    theta = torch.eye(3)
    theta = theta[0:2,:] # slice into submatrix expected by affine_grid
    theta = theta.repeat(N,1,1)
    original_grid = torch.nn.functional.affine_grid(theta, size=grid_shape, align_corners=False)
    if circular: original_grid = wrap_grid_horizontally(original_grid, H, W)
    original_grid[:,:,:,0] = (W-1)*(1 + original_grid[:,:,:,0])/2
    original_grid[:,:,:,1] = (H-1)*(1 + original_grid[:,:,:,1])/2

    # Apply true transformation and its inverse to get final grid
    theta_true = get_transformation(grid_shape, *true_params)
    theta_learned = get_transformation(grid_shape, *learned_params, inverse=True)
    net_theta = theta_learned @ theta_true # apply inverse transformation and get resulting grid
    net_theta = net_theta[0:2,:] # slice into submatrix expected by affine_grid
    net_theta = net_theta.repeat(N,1,1)
    final_grid = torch.nn.functional.affine_grid(net_theta, size=grid_shape, align_corners=False)
    final_grid[:,:,:,0] = (W-1)*(1 + final_grid[:,:,:,0])/2
    final_grid[:,:,:,1] = (H-1)*(1 + final_grid[:,:,:,1])/2
    dist = grid_distance(original_grid, final_grid, IED=IED)
    return dist