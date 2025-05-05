import os
import wfdb

DIR = '/home/joao/Desktop/datasets/grabmyo'

for session_idx in range(3):
    for sub_idx in range(40):
        subses = f"session{session_idx+1}_participant{sub_idx+1}"
        files = os.listdir(os.path.join(DIR, f"Session{session_idx+1}/{subses}"))
        for trial_idx in range(7):
            for gesture_idx in range(17):
                name = f"{subses}_gesture{gesture_idx+1}_trial{trial_idx+1}.hea"
                if name not in files:
                    print(f"Missing {name}")
