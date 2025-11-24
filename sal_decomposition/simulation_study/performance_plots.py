import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


if __name__ == '__main__':
    
    # Load runs
    df = pd.read_csv('gaussian_runs.csv')

    # Parse run name into columns
    df[['Optimization Method', 'SNR', 'Spatial Cutoff Frequency', 'Transform ID']] = df['run_name'].str.split(pat='-', expand=True)
    df['SNR'] = df['SNR'].astype(float)
    df['Spatial Cutoff Frequency'] = df['Spatial Cutoff Frequency'].astype(float)
    df['Transform ID'] = df['Transform ID'].astype(int)

    # Colour pallete chosen
    colours = (["slateblue", "coral", "forestgreen", "goldenrod", "steelblue"])


    # Plot ROAs and MUs kept at different stages
    plt.rcParams['font.size'] = 25
    fig, axs = plt.subplots(3,3, figsize=(28,20))
    fig.subplots_adjust(
    wspace=0.2,  # Horizontal space
    hspace=0.2   # Vertical space
)
    for idx, SNR in enumerate([15, 5, 1]):
        df_snr = df[df['SNR'] == SNR]

        for jdx, opt in enumerate(df['Optimization Method'].unique()):
            df_opt = df_snr[df_snr['Optimization Method'] == opt]
            sns.lineplot(x= df_opt['Spatial Cutoff Frequency'], y=df_opt['summary_rate_of_agreement_refine_avg'], label=opt, ax=axs[0, idx], color=colours[jdx], linewidth=5, marker='o',markersize=20)
            sns.lineplot(x= df_opt['Spatial Cutoff Frequency'], y=df_opt['summary_#mu_matches_refine']/20, label=opt, ax=axs[1, idx], color=colours[jdx], linewidth=5, marker='o', markersize=20)
            sns.lineplot(x= df_opt['Spatial Cutoff Frequency'], y=df_opt['summary_transformation_distance']*0.4, label=opt, color=colours[jdx], ax=axs[2, idx], linewidth=5, marker='o', markersize=20)
        
        # Plot baseline (no optimization)
        sns.lineplot(x= df_snr['Spatial Cutoff Frequency'], y=df_snr['summary_rate_of_agreement_transform_avg'], label='No Optimization', ax=axs[0, idx], color='red', linestyle='--', linewidth=5, marker='o',markersize=20)
        sns.lineplot(x= df_snr['Spatial Cutoff Frequency'], y=df_snr['summary_#mu_matches_transform']/20, label='No Optimization', ax=axs[1, idx], color='red', linestyle='--', linewidth=5, marker='o',markersize=20)

        for metric_idx in range(3):
            axs[metric_idx, idx].set_ylim([-0.1,1.1])
            axs[metric_idx, idx].set_xlim([45,205])
            axs[metric_idx, idx].set_xlabel(None)
            axs[metric_idx, idx].set_ylabel(None)
            axs[metric_idx, idx].set_title(None)
            axs[metric_idx, idx].grid()
            axs[metric_idx, idx].legend([], [], frameon=False)
            axs[metric_idx, idx].set_xticks([62.5, 125.0, 187.5])
            axs[metric_idx, idx].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.savefig('test')
    plt.close('all')

