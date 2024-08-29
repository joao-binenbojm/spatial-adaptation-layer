import os
import json
from tqdm import tqdm

exps = []
for idx in range(1, 6):
    with open(f'conditions/{idx}.json', 'r') as file:
        exps.append(json.load(file))

for exp in tqdm(exps):

    # Save dictionary as current condition to be tested
    with open('exp.json', 'w') as file:
        json.dump(exp, file, indent=4)

    # Run intersession script
    try:
        os.system('python intersession.py') # if you can run the intersession file with the given conditions do so, otherwise move onto the next conditions
    except:
        continue