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
        elif dataset == 'csl':
            exp['emg_tensorizer'] = f"CSLData"
            exp['circular'] = False
            gest_subset = [7,8,9,11,12,13,15]
        elif dataset == 'hyser':
            exp['emg_tensorizer'] = f"HyserData"
            exp['circular'] = False
            gest_subset = [5,6,7,8,9,10,29,30]

    # For each network
    for network in ['LogisticRegressor', 'CapgMyoNet']:
        if network == 'CapgMyoNet':
            exp['num_epochs'] = 2
        elif network == 'LogisticRegressor':
            exp['num_epochs'] = 15
        exp['network'] = network
        for corrective_gain in [True, False]:
            exp['corrective_gain'] = corrective_gain
            for rbase in ["mean-square", "root-mean-square", None]:
                exp['real_baseline'] = rbase
                for median_filter in [True, False]:
                    exp['median-filter'] = median_filter
                    for p_input in [0.0, 0.5]:
                        exp['p_input'] = p_input
                        exp['name'] = f"{dataset}_{network}_{rbase}_{int(median_filter)}_{int(corrective_gain)}"
                        with open(f"sal_classification/{conditions_dir}/{nconditions}.json", 'w') as f:
                            json.dump(exp, f)
                        nconditions += 1
                        print(nconditions)
                        if nconditions % 49 == 0:
                            conditions_dir = conditions_dir[:-1] + str(int(conditions_dir[-1]) + 1) # increases condition dir
                            os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)