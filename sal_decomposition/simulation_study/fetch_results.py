import wandb
import pandas as pd
from tqdm import tqdm


# Define your W&B project details
entity = "jp2717-imperial-college-london"  # Replace with your W&B entity name
project = "sal-decomposition-simulations-freeze"  # Replace with your W&B project name

# Initialize an API object
api = wandb.Api()

# Fetch runs from the specified project
runs = api.runs(f"{entity}/{project}")

# Prepare a list to hold single-row dataframes
single_row_data = []

# Iterate through runs and collect single-row entries
for run in tqdm(runs):
    history = run.history(samples=10000)  # Fetch history with sufficient rows
    if len(history) == 1:  # Check if the history has only one row
        # Add the single row to the list
        single_row_data.append(history.iloc[0])

# Create a DataFrame from the collected single-row entries
df = pd.DataFrame(single_row_data)

# Display or save the resulting DataFrame
print(df)
# Optionally save the DataFrame to a CSV file
df.to_csv("experiments.csv", index=False)