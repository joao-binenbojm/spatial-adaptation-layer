import os

DIR = "/home/joao/Desktop/datasets/hyser2/files/hd-semg/1.0.0/pr_dataset"
# DIR = "/home/joao/Desktop/datasets/hyser/pr_dataset"
subdirs = [d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))]

for subdir in subdirs:
    files = os.listdir(os.path.join(DIR, subdir))
    files = [file for file in files if 'maintenance_raw_sample' in file] # only data files
    files = [file for file in files if '.dat' in file] # only data files
    idxs = [int(file.replace('maintenance_raw_sample','').replace('.dat', '')) for file in files]
    all_idxs = list(range(1, 69))
    print(f"{subdir}:", set(all_idxs).difference(set(idxs)))
    # print(f"{subdir}:", [f for f in files if '.txt' in f])
    # print(any(['.txt' in name for name in files]))


