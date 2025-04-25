import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os


# Function to create column to combine dropout and baseline
dataset='csl' #csl
nruns = 25 #100

## Aggregating all experiment runs into a single DF for joint plotting
exps = pd.read_csv('classification-simulation.csv').sort_values(by='run-name')
exps = exps[exps['dataset'] == dataset]
runnames = exps['run-name']
newdf = pd.DataFrame(np.repeat(exps.values, nruns, axis=0)).reset_index(drop=True)
newdf.columns = exps.columns

df_list = []
for runname in runnames:
    print(runname)
    df_list.append(pd.read_csv(os.path.join('classification-simulation', f'{runname}.csv')))

df_tot = pd.concat(df_list, axis=0).reset_index(drop=True)
df_tot = pd.concat((df_tot, newdf), axis=1)

# Create column to combine dropout and baseline
# df_tot['conditions'] = df_tot.apply(create_intersession_cond_col, axis=1)
df_tot['conditions'] = df_tot['p_input'].astype(str)

# Create new entries with accuracies before any adaptation and after fine-tuning as spatial-adaptaton but different condition so we can plot together
df_no_adapt = df_tot.copy()
df_no_adapt['conditions'] = 'no-adaptation'
df_no_adapt['Tuned Accuracy'] = df_no_adapt['Accuracy']
df_no_adapt['Majority Voting Tuned Accuracy'] = df_no_adapt['Majority Voting Accuracy']
df_no_adapt['Tuned Distance (cm)'] = df_no_adapt['Distance (cm)']
df_no_adapt.drop_duplicates(subset=['p_input', 'dataset', 'real_baseline', 'network', 'conditions'], inplace=True, ignore_index=True)
df_tot = pd.concat((df_no_adapt, df_tot), axis=0).reset_index(drop=True)

# Sort Dataframe to get conditions in the order we want
conditions = ['no-adaptation', '0.0', '0.5']
df_tot['sort'] = df_tot['conditions'].apply(lambda x: {key: idx for idx, key in enumerate(conditions)}[x])
df_tot.sort_values(by='sort', inplace=True, ignore_index=True)

# Plotting
colours = ['midnightblue', 'slateblue', 'cornflowerblue']#, 'mediumpurple', 'lightsteelblue']
plt.figure(figsize=(8,8))
ax= sns.barplot(df_tot, x='network', y='Majority Voting Tuned Accuracy', hue='conditions', palette=colours)
# ax= sns.barplot(df_tot, x='network', y='Tuned Distance (cm)', hue='conditions', palette=colours)

# # Adding texture to fine-tuning bar from barplot
# patterns = ['', '', '', '', '', '', '', '', '', '', '\\', '\\']  # Define patterns for each bar
# for bar, pattern in zip(ax.patches, patterns):
#     bar.set_hatch(pattern)


plt.grid()
# Remove xlabel and ylabel
plt.xlabel(None)  # Removes the xlabel
plt.ylabel(None)  # Removes the ylabel

# Remove legend
plt.legend([], [], frameon=False)  # Hides the legend

plt.ylim([0.0, 1.0])
# plt.show()
plt.savefig(f'C:\\Users\\Joao\\Desktop\\sim_{dataset}.png')