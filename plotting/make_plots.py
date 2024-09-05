import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set_theme(rc={'figure.figsize':(15,12)})
sns.set(font_scale=1.4)

scaling = 24/2

#####################################################################################################
xshifts1 = []
xshifts2 = []
networks = []

# for net in ['Logistic Regressor', 'CapgmyoNet']:
for net in ['Logistic Regressor', 'CapgmyoNet']:
    if net == 'Logistic Regressor':
        df = pd.read_csv('logreg.csv').reset_index(drop=True)
    elif net == 'CapgmyoNet':
        df = pd.read_csv('capgmyonet.csv').reset_index(drop=True)
    for idx in range(5):
        for ses1 in range(5):
            for ses2 in range(ses1+1, 5):
                xshift1 = df.loc[(df['Subjects']==idx)&(df['Train Sessions']==ses1)&(df['Test Sessions']==ses2), 'xshift']
                xshift2 = df.loc[(df['Subjects']==idx)&(df['Train Sessions']==ses2)&(df['Test Sessions']==ses1), 'xshift']
                # xshifts.append((xshift1.tolist()[0]*scaling, xshift2.tolist()[0]*scaling))
                xshifts1.append(xshift1.tolist()[0]*scaling)
                xshifts2.append(xshift2.tolist()[0]*scaling)
                networks.append(net)
    diffs = np.array(xshifts1) - np.array(xshifts2)
    out = ttest_1samp(diffs, popmean=0.0)
    print(f'Model={net} - p-value:', out)



df_plot = pd.DataFrame({'xshift1': xshifts1, 'xshift2': xshifts2, 'Model': networks})
df_plot['Electrode Shift Differences (cm)'] = df_plot['xshift1'] - - df_plot['xshift2']
print(df_plot.groupby(by='Model', as_index=False)['Electrode Shift Differences (cm)'].std())          
# plt.violinplot([x[0]- -x[1] for x in xshifts])
# plt.xlabel('Electrode Shift Mismatch (cm)')
# sns.boxplot(df_plot, x='Model', y='Electrode Shift Differences (cm)')
# plt.axhline(0, ls='--', color='r')
# plt.title('Electrode Shift Deviations Between Alternate Sessions', fontsize=20)
# plt.show()


# ########################################################################
# accs = []
# maj_accs = []
# xshifts = []
# tuned_accs = []
# maj_tuned_accs = []
# networks = []

# df_tot = pd.DataFrame()

# for net in ['CapgmyoNet']:
#     if net == 'Logistic Regressor':
#         df = pd.read_csv('logreg.csv').reset_index(drop=True)
#     elif net == 'CapgmyoNet':
#         df = pd.read_csv('capgmyonet.csv').reset_index(drop=True)
#     # for idx in range(df.shape[0]):
#     #     xshifts.append(df.loc[idx, 'xshift'])
#     #     accs.apppend(df.loc[idx, 'Accuracy'])
#     #     maj_accs.append(df.loc[idx, 'Majority Voting Accuracy'])
#     #     tuned_accs.append(df.loc[idx, 'Tuned Accuracy'])
#     #     maj_tuned_accs.append(df.loc[idx, 'Majority Voting Tuned Accuracy'])
#     networks.extend([net]*df.shape[0])
#     df_tot = pd.concat((df_tot, df), ignore_index=True)


# df_tot['MJ Acc. Improvement (%)'] = (df_tot['Majority Voting Tuned Accuracy'] - df_tot['Majority Voting Accuracy']) / df_tot['Majority Voting Accuracy']
# df_tot['Acc. Improvement'] = df_tot['Tuned Accuracy'] - df_tot['Accuracy']
# df_tot['Model'] = networks
# df_tot['Circumferential Shift Magnitude (cm)'] = scaling*df_tot['xshift'].abs()
# print(df_tot)

# plt.figure()
# sns.regplot(df_tot, x='Circumferential Shift Magnitude (cm)', y='MJ Acc. Improvement (%)')
# plt.show()



