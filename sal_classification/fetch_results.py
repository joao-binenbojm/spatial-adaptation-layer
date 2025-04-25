import wandb
import os
import pandas as pd
import json

# Define the project and entity details
PROJECT_NAME = "classification-simulation"  # Replace with your project name
ENTITY = "jp2717-imperial-college-london"  # Replace with your entity name (team or username)
OUTPUT_DIR = "classification-simulation"  # Directory to save the CSV files

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_intrasession_conditions_csv():
    """
    Connects to the specified Weights & Biases project, retrieves runs, and saves each run as a row of a .csv dataframe.
    """ 
    # Initialize the wandb API
    api = wandb.Api()

    # Get the list of runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

    print(f"Found {len(runs)} runs in project '{PROJECT_NAME}'.")

    # Iterate through the runs
    keys = ['dataset', 'network', 'p_input', 'real_baseline']
    run_names = []
    data = {key: [] for key in keys}
    for run in runs:
        run_names.append(run.name or run.id)  # Use run name or ID as a fallback
        print(f"Processing run: {run_names[-1]}")
        for key in data.keys(): data[key].append(run.config[key])
    
    df = pd.DataFrame(data)
    df['run-name'] = run_names
    df.to_csv('intrasession.csv', index=False)


def save_project_wandb_tables_as_csv():
    """
    Connects to the specified Weights & Biases project, retrieves runs, and saves wandb.Table objects as CSVs.
    """
    # Initialize the wandb API
    api = wandb.Api()

    # Get the list of runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

    print(f"Found {len(runs)} runs in project '{PROJECT_NAME}'.")

    # Iterate through the runs
    for run in runs:
        run_name = run.name or run.id  # Use run name or ID as a fallback
        print(f"Processing run: {run_name}")

        try:
            for artifact in run.logged_artifacts():
                if 'complete_results' in artifact.name:
                    artifact_dir = artifact.download()
                    with open(os.path.join(artifact_dir, 'complete_results.table.json'), 'rb') as f:
                        res = json.load(f)
                    cols, data = res['columns'], res['data']
                    df = pd.DataFrame(columns=cols, data=data)
                    df.to_csv(os.path.join(OUTPUT_DIR, f'{run_name}.csv'), index=False)

        except Exception as e:
            print(f"Error saving table from run '{run_name}': {e}")


def get_intersession_conditions_csv():
    """
    Connects to the specified Weights & Biases project, retrieves runs, and saves each run as a row of a .csv dataframe.
    """ 
    # Initialize the wandb API
    api = wandb.Api()

    # Get the list of runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

    print(f"Found {len(runs)} runs in project '{PROJECT_NAME}'.")

    # Iterate through the runs
    keys = ['dataset', 'network', 'p_input', 'real_baseline', 'learnable_baseline', 'adaptation']
    run_names = []
    data = {key: [] for key in keys}
    for run in runs:
        run_names.append(run.name or run.id)  # Use run name or ID as a fallback
        print(f"Processing run: {run_names[-1]}")
        for key in data.keys(): data[key].append(run.config[key])
    
    df = pd.DataFrame(data)
    df['run-name'] = run_names
    df.to_csv('intersession.csv', index=False)


def get_simulation_conditions_csv():
    """
    Connects to the specified Weights & Biases project, retrieves runs, and saves each run as a row of a .csv dataframe.
    """ 
    # Initialize the wandb API
    api = wandb.Api()

    # Get the list of runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

    print(f"Found {len(runs)} runs in project '{PROJECT_NAME}'.")

    # Iterate through the runs
    keys = ['dataset', 'network', 'p_input', 'real_baseline', 'learnable_baseline', 'adaptation']
    run_names = []
    data = {key: [] for key in keys}
    for run in runs:
        run_names.append(run.name or run.id)  # Use run name or ID as a fallback
        print(f"Processing run: {run_names[-1]}")
        for key in data.keys(): data[key].append(run.config[key])
    
    df = pd.DataFrame(data)
    df['run-name'] = run_names
    df.to_csv('simulation.csv', index=False)


if __name__ == "__main__":
    save_project_wandb_tables_as_csv()
    get_simulation_conditions_csv()