import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('bandpass.csv')
    df['filter'] = ['bandpass']*18
    df2 = pd.read_csv('bandstop.csv')
    df2['filter'] = ['bandstop']*18
    df = pd.concat((df, df2))
    plt.figure()
    sns.violinplot(data=df, x='filter', y='Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot.png')
    plt.figure()
    sns.violinplot(data=df, x='filter', y='MV Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot2.png')

    print('Test Accuracies:')
    print(df.groupby(by='filter').mean())
