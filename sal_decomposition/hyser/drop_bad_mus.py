import pandas as pd
import pickle
import json
import numpy as np
from sal_decomposition.utils import utils
import torch

file = "decomposition_data_subject2_session2.pkl"
with open(f"sal_decomposition/hyser/{file}", 'rb') as f:
    data_dict = pickle.load(f)

## ASSUMING WE USE DEBUGGER TO DROP THE MUs WE DON'T LIKE
print()

with open(f"sal_decomposition/hyser/_{file}", 'wb') as f:
    # 3. Dump the object into the file
    pickle.dump(data_dict, f)