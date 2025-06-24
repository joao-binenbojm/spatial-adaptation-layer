import os
import sys
import json
from tqdm import tqdm


# filenames = os.listdir('./sal_classification/conditions1')
# for name in filenames:
filenames = os.listdir('./sal_classification/conditions1')
for idx in range(len(filenames)):
    try:
        os.system(f"python ./sal_classification/intersession_new.py conditions1/{idx+1}") # if you can run the intersession file with the given conditions do so, otherwise move onto the next conditions
    except:
        continue
