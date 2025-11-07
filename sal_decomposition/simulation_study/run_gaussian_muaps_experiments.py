import json
import os
import sys
from tqdm import tqdm

exp_names = os.listdir('./sal_decomposition/simulation_study/exp_conditions')
exp_names = ['exp_conditions/' + name.replace('.json', '') for name in exp_names if name.endswith('.json')]
failed_conditions = []

for exp_name in tqdm(exp_names):
    print(exp_name)
    exit_code = os.system(f"python3 sal_decomposition/simulation_study/experiment_gaussian_muaps.py {exp_name}")
    if exit_code != 0:
        print(f"Experiment {exp_name} failed with exit code {exit_code}")
        failed_conditions.append(exp_name)

print("Failed conditions:", failed_conditions)
