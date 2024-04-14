### THIS SCRIPT IS USED TO CONVERT THE FILE ARRANGEMENT OF THE CAPGMYO DATASET TO THE CSL TO HAVE MORE UNIFORM DATA LOADING
import numpy as np
import os
from tqdm import tqdm
import shutil

def force_mkdir(path):
    '''Checks if directory exists and makes if it doesn't, remakes another if it does.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)

def copy_and_rename(src_path, dest_path, old_name, new_name):
    # Copy the file
    shutil.copy(os.path.join(src_path, old_name), dest_path)
 
    # Rename the copied file
    old = f"{dest_path}/{old_name}"
    new = f"{dest_path}/{new_name}"
    os.rename(old, new)


if __name__ == '__main__':
    DIR = '../datasets/capgmyo/dbb'
    names = os.listdir(DIR)
    SAVE_DIR = '../datasets/capgmyo/dbb_csl'
    force_mkdir(SAVE_DIR)

    for idx in tqdm(range(1, 19, 2)): # for each subject
        SUB_DIR = '{}/subject{}'.format(SAVE_DIR, idx//2 + 1) 
        force_mkdir(SUB_DIR)

        for sdx in range(2): # for each session
            S_DIR = '{}/subject{}/session{}'.format(SAVE_DIR, idx//2 + 1, sdx + 1)
            force_mkdir(S_DIR)

            # Get appropriate filenames for each condition
            sub_id = '00'+str(idx + sdx) if (idx + sdx) < 10 else '0'+str(idx + sdx)
            sub_names = filter(lambda name: name[:3]==sub_id, names)

            for name in sub_names:
                if name[4] != '1': # if gesture file and not a contraction force file
                    gest = int(name[4:7].lstrip('0'))
                    new_name = 'gest{}.mat'.format(gest)
                    copy_and_rename(DIR, S_DIR, name, new_name) # copy file and rename


    