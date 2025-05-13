import json
import os

with open('sal_classification/exp-base.json', 'rb') as f:
    exp = json.load(f)

conditions_dir = 'conditions1'
os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)

nconditions = 1
exp['network'] = 'CapgMyoNet'
for num_epochs in [1, 5, 10]:
    exp['num_epochs'] = num_epochs
    for rbase in [True, False]:
        exp['real-baseline'] = rbase
        for median_filter in [True, False]:
            exp['median-filter'] = median_filter
            for p_input in [0.0, 0.25, 0.5, 0.75]:
                exp['p_input'] = p_input
                exp['name'] = f"cnn_{num_epochs}_{int(rbase)}_{int(median_filter)}_{p_input}"
                with open(f"sal_classification/{conditions_dir}/{nconditions}.json", 'w') as f:
                    json.dump(exp, f)
                nconditions += 1
                if nconditions % 49 == 0:
                    conditions_dir = conditions_dir[:-1] + str(int(conditions_dir[-1]) + 1) # increases condition dir
                    os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)




exp['network'] = 'LogisticRegressor'
for num_epochs in [15, 30, 50]:
    exp['num_epochs'] = num_epochs
    for rbase in [True, False]:
        exp['real-baseline'] = rbase
        for median_filter in [True, False]:
            exp['median-filter'] = median_filter
            for p_input in [0.0, 0.25, 0.5, 0.75]:
                exp['p_input'] = p_input
                exp['name'] = f"logreg_{num_epochs}_{int(rbase)}_{int(median_filter)}_{p_input}"
                with open(f"sal_classification/{conditions_dir}/{nconditions}.json", 'w') as f:
                    json.dump(exp, f)
                nconditions += 1
                if nconditions % 49 == 0:
                    conditions_dir = conditions_dir[:-1] + str(int(conditions_dir[-1]) + 1) # increases condition dir
                    os.makedirs(f"sal_classification/{conditions_dir}", exist_ok=True)


