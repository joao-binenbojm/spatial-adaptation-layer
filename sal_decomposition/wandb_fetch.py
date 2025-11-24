import pandas as pd
import numpy as np
import wandb
from collections import Counter
import json
from tqdm import tqdm

# 1. Initialize the W&B Public API
api = wandb.Api()

# 2. Define your entity and project name
entity_name = "jp2717-imperial-college-london"
project_name = "sda-real-data-simulations2"
project_path = f"{entity_name}/{project_name}"

# 3. Retrieve the runs
runs = api.runs(path=project_path, lazy=False)

# Initialize a list to hold all run data
all_runs_data = []

print(f"Fetching runs from {project_path}...")

# 4. Iterate through runs and extract data
for run in tqdm(runs):
    # Start with basic metadata
    run_data = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state
    }

    # Extract config (hyperparameters)
    # We prefix with 'config_' to avoid name collisions with summary metrics
    # config = json.loads(run.config)
    for k, v in run.config.items():
        run_data[f"config_{k}"] = v

    # Extract summary (results like loss, accuracy)
    # We prefix with 'summary_' and filter out internal wandb keys (start with _)
    for k, v in run.summary.items():
        if not k.startswith("_"):
            run_data[f"summary_{k}"] = v

    all_runs_data.append(run_data)

# 5. Create a DataFrame and save to CSV
df = pd.DataFrame(all_runs_data)
output_filename = "runs.csv"

df.to_csv(output_filename, index=False)

print(f"\nSaved {len(df)} runs to '{output_filename}'")
print(f"Columns collected: {len(df.columns)}")
