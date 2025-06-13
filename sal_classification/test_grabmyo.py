import pandas
import wfdb
import numpy as np
import os


DIR = "/home/joao/Desktop/datasets/grabmyo"

# TESTING WHICH FILES ARE MISSING FROM DATASETS DIRECTORY FOR GRABMYO
for sesidx in range(3):
    for subidx in range(43):
        subsesdir = os.path.join(DIR, f"Session{sesidx+1}/session{sesidx+1}_participant{subidx+1}")
        subsesfiles = os.listdir(subsesdir)
        for gest_idx in range(17):
            for trial_idx in range(7):
                recname = f"session{sesidx+1}_participant{subidx+1}_gesture{gest_idx+1}_trial{trial_idx+1}"
                if recname + ".dat" not in subsesfiles:
                    print(recname + ".dat")
                if recname + '.hea' not in subsesfiles:
                    print(recname + ".hea")