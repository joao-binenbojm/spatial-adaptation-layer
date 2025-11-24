import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


if __name__ == '__main__':
    
    # Load runs
    df = pd.read_csv('perturbations_runs.csv')

    # Parse run name into columns
    # df[['Optimization Method', 'SNR', 'Spatial Cutoff Frequency', 'Transform ID']] = df['run_name'].str.split(pat='-', expand=True)
    # df['SNR'] = df['SNR'].astype(float)
    # df['Spatial Cutoff Frequency'] = df['Spatial Cutoff Frequency'].astype(float)
    # df['Transform ID'] = df['Transform ID'].astype(int)

    initial_mu_counts = {'1-1-25': 69, '1-2-25': 59, '1-3-25': 57,
                         '1-1-50': 64, '1-2-50': 54, '1-3-50': 50,
                         '2-1-25': 45, '2-2-25': 47, '2-3-25': 33,
                         '2-1-50': 32, '2-2-50': 32, '2-3-50': 23,
                         '3-1-25': 26, '3-2-25': 40, '3-3-25': 41,
                         '3-1-50': 18, '3-2-50': 34, '3-3-50': 38
            }
    def normalize_by_mu_count(row, col_names):
        key = f"{row['config_Subject']}-{row['config_Session']}-{row['config_MVC']}"
        initial_count = initial_mu_counts.get(key, 1)
        for col_name in col_names:
            row[col_name] = row[col_name] / initial_count
        return row

    col_names = ["summary_#mu_matches", "summary_#mu_matches_base", "summary_#mu_matches_refine", "summary_#mu_matches_transform"]
    df = df.apply(lambda row: normalize_by_mu_count(row, col_names), axis=1)

    # Colour pallete chosen
    colours = (["slateblue", "coral", "forestgreen", "goldenrod", "steelblue"])


    # Plot ROAs and MUs kept at different stages
    plt.rcParams['font.size'] = 25
    fig, axs = plt.subplots(3,2, figsize=(28,20))
    fig.subplots_adjust(
    wspace=0.2,  # Horizontal space
    hspace=0.2   # Vertical space
)
    for idx, mvc in enumerate([25, 50]):
        df_mvc = df[df['config_MVC'] == mvc]

        # for jdx, opt in enumerate(df['config_opt'].unique()):
        sns.boxplot(data=df_mvc, x='config_opt', y='summary_rate_of_agreement_refine_avg', ax=axs[0, idx], palette=colours)
        sns.boxplot(data=df_mvc, x='config_opt', y='summary_#mu_matches_refine', ax=axs[1, idx], palette=colours)
        sns.boxplot(data=df_mvc, x='config_opt', y='summary_transformation_distance', ax=axs[2, idx], palette=colours)#', linewidth=5, marker='o',markersize=20)

        # sns.boxplot(x= df_opt['Spatial Cutoff Frequency'], y=df_opt['summary_transformation_distance']*0.4, label=opt, color=colours[jdx], ax=axs[2, idx], linewidth=5, marker='o', markersize=20)
        
        # # Plot baseline (no optimization)
        # sns.lineplot(x= df_snr['Spatial Cutoff Frequency'], y=df_snr['summary_rate_of_agreement_transform_avg'], label='No Optimization', ax=axs[0, idx], color='red', linestyle='--', linewidth=5, marker='o',markersize=20)
        # sns.lineplot(x= df_snr['Spatial Cutoff Frequency'], y=df_snr['summary_#mu_matches_transform']/20, label='No Optimization', ax=axs[1, idx], color='red', linestyle='--', linewidth=5, marker='o',markersize=20)

        for metric_idx in range(3):
            axs[metric_idx, idx].set_ylim([-0.1,1.1])
            axs[metric_idx, idx].set_xlabel(None)
            axs[metric_idx, idx].set_ylabel(None)
            axs[metric_idx, idx].set_title(None)
            axs[metric_idx, idx].grid()
            axs[metric_idx, idx].legend([], [], frameon=False)
            axs[metric_idx, idx].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.savefig('performance_plots_perturbations.png')
    plt.close('all')

