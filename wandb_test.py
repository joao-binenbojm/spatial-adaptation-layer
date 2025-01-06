import wandb

# Set your project name and entity (user/organization)
project_name = 'jp2717-imperial-college-london'
entity_name = 'sal-decomposition-simulations3'

# Initialize the API
api = wandb.Api()

# Get the project
project = api.project(f"{entity_name}/{project_name}")

# Loop through and delete all runs
for run in project.runs():
    print(f"Deleting run: {run.id}")
    run.delete()