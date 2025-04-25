import wfdb
import os
import matplotlib.pyplot as plt
from emg_processing import get_rms_signal

Mrms = int(0.25*2048)


DIR = "/home/joao/Desktop/datasets/grabmyo/Session1/session1_participant1"
files = os.listdir(DIR)
fnames = [n.replace('.hea','').replace('.dat','') for n in files]
fnames = list(set(fnames))

t = 5 # just for a random trial to double check
variance = []
for idx in range(17):
# for f in fnames:
    f = f'session1_participant1_gesture{idx+1}_trial{t}'
    record = wfdb.rdrecord(os.path.join(DIR, f))
    print(f'Gesture #{idx+1} - Var={record.p_signal.var()}')
    variance.append(record.p_signal.var())
    # plt.figure()
    # plt.plot(rms)
    # plt.savefig('rms')
    # print()

import numpy as np
print(np.argmin(variance))
plt.figure()
plt.stem(variance)
plt.savefig('gest_var')    

