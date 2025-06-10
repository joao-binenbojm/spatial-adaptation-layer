import os
import sys
import json
from tqdm import tqdm


filenames = os.listdir('./sal_classification/conditions3')
for name in filenames:
    try:
        os.system(f"python ./sal_classification/intersession_new.py conditions3/{name.replace('.json','')}") # if you can run the intersession file with the given conditions do so, otherwise move onto the next conditions
    except:
        continue
