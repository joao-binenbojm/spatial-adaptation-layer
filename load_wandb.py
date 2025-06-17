import wandb
import pandas as pd
import os
import json
import tqdm

for conditions_dir in ['conditions1', 'conditions2']:
    for idx in tqdm.tqdm(range(1, 21)):
        with open(f"sal_classification/{conditions_dir}/{idx}.json") as f:
            config = json.load(f)
            config['scheduler'] = json.dumps(config['scheduler'])
            name = config['name']
            df = pd.read_csv(f'{name}.csv')

        # Log wandb conditions
        wandb.init(
            # set the wandb project where this run will be logged
            project=config["project"],
            config=config,
            name=name
            # mode='disabled'
        )

        table = wandb.Table(dataframe=df)
        wandb.log({'complete_results': table})
        # wandb.log({'performance_histogram': wandb.plot.histogram(table, "Majority Voting Tuned Accuracy",
        #   title="Performance Distribution Across Dataset")})
        wandb.log({'Accuracy': df['Accuracy'].mean()})
        wandb.log({'Tuned Accuracy': df['Tuned Accuracy'].mean()})

        wandb.finish()
