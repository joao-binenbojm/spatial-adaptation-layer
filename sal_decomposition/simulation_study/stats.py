import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# from sal_decomposition.simulation_study.sda_pipeline import SDAExperiment
import torch 
from tqdm import tqdm


def get_transformed_grid(grid_shape, Tx=0, Ty=0, theta=0, xscale=1, yscale=1):
    '''Computes the transformed grid coordinates for euclidina distance comparison.'''

    N, C, H, W = grid_shape
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

    # Obtain transformed grid in pixel units
    grid = torch.nn.functional.affine_grid(theta, size = (N,C,H, W), align_corners=False)
    grid[:,:,:,0] = W*(1 + grid[:,:,:,0])/2
    grid[:,:,:,1] = H*(1 + grid[:,:,:,1])/2
    return grid

def grid_distance(grid1, grid2, IED=4):
    '''Computes Euclidian distance between grid coordinates of true and learned transformations. Returns distance in mm.'''
    dist = torch.sqrt((grid1*IED - grid2*IED)**2).mean()
    return dist.item()

if __name__ == '__main__':
    grid_shape = (1,1,25,10)
    df = pd.read_csv('sal_decomposition/simulation_study/experiments_freeze.csv')
    # df2 = pd.read_csv('sal_decomposition/simulation_study/experiments_freeze.csv')
    
    # # Filter out SNRs
    # df = df[df['SNR'] < 5].reset_index(drop=True)
    # df2 = df2[df2['SNR'] < 5].reset_index(drop=True)

    # # Compute average sensitivity difference
    # df = df.groupby(by=['SNR', 'fxmax', 'opt'], as_index=False)['sensitivity_avg'].mean()
    # df2 = df2.groupby(by=['SNR', 'fxmax', 'opt'], as_index=False)['sensitivity_avg'].mean()
    # df2.rename(columns={'sensitivity_avg': 'sensitivity_avg2'}, inplace=True)
    # df.sort_values(axis='index', by=['SNR', 'fxmax', 'opt'], inplace=True)
    # df2.sort_values(axis='index', by=['SNR', 'fxmax', 'opt'], inplace=True)

    # df_join = pd.concat((df, df2[['sensitivity_avg2']]), axis=1)
    # df_join['diff'] = df_join['sensitivity_avg'] - df_join['sensitivity_avg2']
    # df_join.drop(columns=['sensitivity_avg', 'sensitivity_avg2'], inplace=True)
    # print(df_join)

    # print(df_join.groupby(by=['opt', 'fxmax'])['diff'].mean())

    
    # for SNR in [30, 15, 5, 1]:
        # for fxmax in [62.5, 93.75, 125, 156.25, 187.5]:

    dists = []
    df = df[df['SNR'] == 15].reset_index(drop=True)
    for idx in tqdm(range(df.shape[0])):
        Tx, Ty, theta, xscale, yscale = df.loc[idx, ['Tx', 'Ty', 'theta', 'xscale', 'yscale']]
        Tx_opt, Ty_opt, theta_opt, xscale_opt, yscale_opt = df.loc[idx, ['Tx_opt', 'Ty_opt', 'theta_opt', 'xscale_opt', 'yscale_opt']]
        Tx_opt, Ty_opt, theta_opt, xscale_opt, yscale_opt = -Tx_opt, -Ty_opt, -theta_opt, 1/xscale_opt, 1/yscale_opt

        # Get ground truth grid
        grid = get_transformed_grid(grid_shape, Tx, Ty, theta, xscale, yscale)
        grid_opt = get_transformed_grid(grid_shape, Tx_opt, Ty_opt, theta_opt, xscale_opt, yscale_opt)

        dist = grid_distance(grid, grid_opt)
        dists.append(dist)
        # print(f'Displacement of {dist} mm')
        # print(df.loc[idx, 'opt'])
        # print(Tx_opt, Ty_opt, theta_opt, xscale_opt, yscale_opt)
        # print(Tx, Ty, theta, xscale, yscale)

    df['Distance (mm)'] = dists

    # Add original onto opt column
    df2 = df.copy()
    df2['sensitivity_avg'] = df2['sensitivity_base_avg']
    df2['precision_avg'] = df2['precision_base_avg']
    df2['opt'] = 'optimal'
    df2.drop_duplicates(subset=['SNR', 'fxmax', 'opt'], inplace=True)
    
    # Add transform column onto opt column
    df3 = df.copy()
    df3['sensitivity_avg'] = df3['sensitivity_transform_avg']
    df3['precision_avg'] = df3['precision_transform_avg']
    df3['opt'] = 'post_transform'
    df3.drop_duplicates(subset=['SNR', 'fxmax', 'opt', 'Tx', 'Ty', 'theta', 'xscale', 'yscale'], inplace=True)

    df = pd.concat((df, df2, df3), axis=0).reset_index(drop=True)
    df.drop_duplicates(subset=['SNR', 'opt', 'fxmax'])

    palette = ['lightsalmon', 'navajowhite', 'darkgreen', 'teal', 'steelblue']
    # palette = ['lightsalmon', 'navajowhite', 'darkgreen']

    plt.figure()
    # sns.pointplot(df, x='fxmax', y='Distance (mm)', hue='opt')
    # sns.pointplot(df, x='fxmax', y='sensitivity_avg', hue='opt', linestyles=['-', '-', '-', '--', '--'], palette='rocket_r')
    sns.pointplot(df, x='fxmax', y='precision_avg', hue='opt', linestyles=['-', '-', '-', '--', '--'], palette=palette, scale=2)
    # sns.pointplot(df, x='fxmax', y='Distance (mm)', hue='opt', linestyles=['-', '-', '-'], palette=palette, scale=2)
    # # sns.pointplot(df, x='fxmax', y='sensitivity_base_avg', linestyles='--')
    plt.grid()
    plt.ylim([0.0, 1.0])
    
    # Remove xlabel and ylabel
    plt.xlabel(None)  # Removes the xlabel
    plt.ylabel(None)  # Removes the ylabel

    # Remove legend
    plt.legend([], [], frameon=False)  # Hides the legend

    plt.savefig('test_plot')
    # init_dist = np.sqrt((df['Tx']/3)**2 +  (df['Ty']/3)**2 + (df['theta']/np.pi)**2 + ((df['xscale']-1)/0.2)**2 + ((df['yscale']-1)/0.2)**2)
    # df['init_dist'] = init_dist
    # df = df[df['opt'] == 'search'].reset_index()
    # plt.figure()
    # sns.scatterplot(df, x='init_dist', y='sensitivity_avg', hue='fxmax')
    # plt.savefig('test_plot')
    
    # # Compute the dists vs sensitivity
    # df = df[(df['opt'] == 'search_fit')].reset_index(drop=True)
    # plt.figure()
    # sns.scatterplot(df, x='Distance (mm)', y='sensitivity_avg')
    # plt.savefig('perf_vs_dists.jpg')

    # EXPONENTIAL FIT

    # Linearized model
    # log_b = np.log(df['sensitivity_avg'])
    X = sm.add_constant(df['Distance (mm)'])  # Add intercept for the regression
    model = sm.OLS(df['sensitivity_avg'], X).fit()
    b0, k = model.params  # Extract parameters (b0_log is log(b0))
    # b0 = np.exp(b0_log)  # Convert log(b0) back to b0

    # Generate fitted curve
    df['sensitivity_avg_fit'] = b0  + k*df['Distance (mm)']

    # Plot with seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Distance (mm)', y='sensitivity_avg', data=df, label='Data', color='blue', alpha=0.7)
    sns.lineplot(x='Distance (mm)', y='sensitivity_avg_fit', data=df, label='Exponential Fit', color='red', linewidth=2)
    plt.xlabel('Distance (mm)')
    plt.ylabel('Sensitivity')
    plt.title('Exponential Decay Fit')
    plt.legend()

    # Annotate p-value
    p_value = model.pvalues[1]  # p-value for the decay rate
    print(p_value)
    plt.text(0.05, 0.95, f'$p$-value: {p_value:.3e}', transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.grid()
    plt.savefig('exponential_fit.jpg')
    