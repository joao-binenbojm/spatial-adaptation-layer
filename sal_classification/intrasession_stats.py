import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os


# Function to create column to combine dropout and baseline
def create_intrasession_cond_col(row):
    if row['p_input'] and row['real_baseline']:
        return 'BN+Drpt.'
    elif row['p_input'] and not row['real_baseline']:
        return 'Drpt.'
    elif not row['p_input'] and row['real_baseline']:
        return 'BN'
    else:
        return 'Base'

dataset='csl' #capgmyo
nruns = 25 #16

## Aggregating all experiment runs into a single DF for joint plotting
exps = pd.read_csv('intrasession.csv').sort_values(by='run-name')
exps = exps[exps['dataset'] == dataset]
runnames = exps['run-name']
newdf = pd.DataFrame(np.repeat(exps.values, nruns, axis=0)).reset_index(drop=True)
newdf.columns = exps.columns

df_list = []
for runname in runnames:
    print(runname)
    df_list.append(pd.read_csv(os.path.join('intrasession', f'{runname}.csv')))

df_tot = pd.concat(df_list, axis=0).reset_index(drop=True)
df_tot = pd.concat((df_tot, newdf), axis=1)

# Create column to combine dropout and baseline
df_tot['conditions'] = df_tot.apply(create_intrasession_cond_col, axis=1)

colours = ['slateblue', 'cornflowerblue', 'midnightblue', 'lightsteelblue']
# df_tot = df_tot[df_tot['dataset'] == 'capgmyo'].reset_index(drop=True)
plt.figure(figsize=(8,8))
sns.barplot(df_tot, x='network', y='Majority Voting Accuracy', hue='conditions', palette=colours)
plt.grid()
# Remove xlabel and ylabel
plt.xlabel(None)  # Removes the xlabel
plt.ylabel(None)  # Removes the ylabel

# Remove legend
# plt.legend([], [], frameon=False)  # Hides the legend

plt.ylim([0.0, 1.0])
plt.show()
# plt.savefig(f'C:\\Users\\Joao\\Desktop\\intra_{dataset}.png')

print()