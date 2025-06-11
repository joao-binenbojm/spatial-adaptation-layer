import wandb

api = wandb.Api()

# Replace with your entity (username or team) and project name
entity = "jp2717-imperial-college-london"
project = "intersession-new"

runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(f"Deleting run: {run.name} ({run.id})")
    run.delete()