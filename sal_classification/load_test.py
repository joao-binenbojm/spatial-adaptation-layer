import wfdb
import os

DIR = '../datasets/hyser/pr_dataset/subject01_session1'
record = wfdb.rdrecord(os.path.join(DIR, 'maintenance_raw_sample1'))  # Just the base name, no .hea or .dat
print()