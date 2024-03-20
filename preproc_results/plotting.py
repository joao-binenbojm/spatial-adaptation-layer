import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('no_preproc.csv')
    df['proc'] = ['no_proc']*18
    df2 = pd.read_csv('standardize.csv')
    df2['proc'] = ['standardize']*18
    df3 = pd.read_csv('minmax.csv')
    df3['proc'] = ['minmax']*18
    df = pd.concat((df, df2, df3))
    plt.figure()
    sns.violinplot(data=df, x='proc', y='Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot.png')
    plt.figure()
    sns.violinplot(data=df, x='proc', y='MV Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot2.png')

    print('Test Accuracies:')
    print(df.groupby(by='proc').mean())
