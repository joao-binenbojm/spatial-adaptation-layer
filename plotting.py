import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('maj.csv')
    df['mj'] = ['1000']*9
    df2 = pd.read_csv('bandstop_inter.csv')
    df2['mj'] = ['32']*9
    df = pd.concat((df, df2))
    plt.figure()
    sns.violinplot(data=df, x='mj', y='Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot.png')
    plt.figure()
    sns.violinplot(data=df, x='mj', y='MV Accuracy')
    plt.title('Comparing outcomes of different preprocessing methods')
    plt.savefig('plot2.png')

    print('Test Accuracies:')
    print(df.groupby(by='mj').mean())
