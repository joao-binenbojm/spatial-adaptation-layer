import pandas as pd


df = pd.read_csv("runs.csv")
df = df[df['config_opt'] == "search_fit"].reset_index(drop=True)
print(df['run_name'].value_counts())