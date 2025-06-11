import json
import os

with open('sal_classification/exp-base.json', 'rb') as f:
    exp = json.load(f)

conditions_dir = 'conditions1'
os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)

nconditions = 1

# For each dataset
for dataset in ['csl', 'hyser', 'capgmyo', 'grabmyo-forearm', 'grabmyo-wrist']:
    exp['dataset'] = dataset
    if 'grabmyo' in dataset:
        exp['median-filter'] = False
        exp['real_baseline'] = "mean-square"
        for key in exp['adaptation_params'].keys():
            if key == 'xshift':
                exp['adaptation_params'][key] = True
            else:
                exp['adaptation_params'][key] = False
        exp['circular'] = True
        exp['emg_tensorizer'] = 'GrabmyoData'
        exp['gest_subset'] = [10,11,12,13,14,15]
    else:
        for key in exp['adaptation_params'].keys(): exp['adaptation_params'][key] = True
    
        if dataset == 'capgmyo':
            exp['emg_tensorizer'] = f"CapgmyoData"
            exp['circular'] = True
            exp['gest_subset'] = None
            exp['real_baseline'] = "mean-square"
            exp['median-filter'] = False
        elif dataset == 'csl':
            exp['emg_tensorizer'] = f"CSLData"
            exp['circular'] = False
            exp['gest_subset'] = [7,8,11,12,15,20,22,23]
            exp['real_baseline'] = "mean_square"
            exp['median-filter'] = True
        elif dataset == 'hyser':
            exp['emg_tensorizer'] = f"HyserData"
            exp['circular'] = False
            exp['real_baseline'] = None
            exp['gest_subset'] = [5,6,7,8,9,10,29,30]
            exp['median-filter'] = True
    # For each adaptation method
    for adaptation in ['spatial-adaptation', 'fine-tuning', 'linear-layer', 'scratch-training', 'adabatch']:
        exp['adaptation'] == adaptation
        # For each network
        for network in ['LogisticRegressor', 'CapgMyoNet']:
            if network == 'CapgMyoNet':
                exp['num_epochs'] = 2
                exp['p_input'] = 0.0
            elif network == 'LogisticRegressor':
                exp['num_epochs'] = 15
                exp['p_input'] = 0.5 * int("grabmyo" not in dataset)
            exp['network'] = network
            exp['name'] = f"{dataset}_{adaptation}_{network}"
            with open(f"sal_classification/{conditions_dir}/{nconditions}.json", 'w') as f:
                json.dump(exp, f)
            nconditions += 1
            print(nconditions)
            if nconditions % 49 == 0:
                conditions_dir = conditions_dir[:-1] + str(int(conditions_dir[-1]) + 1) # increases condition dir
                os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)