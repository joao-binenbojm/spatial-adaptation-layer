import wfdb
import os
import matplotlib.pyplot as plt
from emg_processing import get_rms_signal

Mrms = int(0.25*2048)


DIR = r"C:\Users\Joao\Desktop\datasets\grabmyo\Session1\session1_participant1"
files = os.listdir(DIR)
fnames = [n.replace('.hea','').replace('.dat','') for n in files]
fnames = list(set(fnames))

for f in fnames:
    record = wfdb.rdrecord(os.path.join(DIR, f))
    rms = get_rms_signal(record.p_signal, Mrms)
    plt.figure()
    plt.plot(rms)
    plt.savefig('rms')
    print()


