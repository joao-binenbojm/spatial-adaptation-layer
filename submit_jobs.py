import os
import json
import subprocess

# Predefine the amount of time allocated to each job depending on the job (in hours)
walltimes = {'LogisticRegressor': {15:1, 30:3, 50:5}, 'CapgMyoNet':{1:2, 5:10, 10:20 } }

def load_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # data['filename'] = filename
                json_data.append(data)
    return json_data

def generate_job_script(walltime):
    job_script = "#!/bin/bash\n"
    job_script += "#PBS -l select=1:ncpus=4:mem=30gb:ngpus=1\n"
    job_script += f"#PBS -l walltime={walltime}:00:00\n\n"
    job_script += "cd $PBS_O_WORKDIR\n\n"
    job_script += "module load anaconda3/personal\n"
    job_script += "source activate meta-adabatch\n"
    job_script += "cd spatial-adaptation-layer\n\n"
    job_script += f"PYTHONPATH=\"$(pwd)\" python3 ./sal_classification/intrasession_new.py tmp \n"
    return job_script

def submit_jobs(json_files):
    tmp_job_script = "tmp_job.sh"
    for json_data in json_files:
        try:
            walltime = walltimes[json_data['network']][json_data['num_epochs']]
        except:
            walltime = 10 # 10h by default

        walltime = f'0{walltime}' if walltime < 10 else str(walltime)
        job_script = generate_job_script(walltime=walltime)
        
        # with open('spatial-adaptation-layer/sal_classification/tmp.json', 'w') as f:
        with open('sal_classification/tmp.json', 'w') as f:
            json.dump(json_data, f)

        # Write the temporary job script
        with open(tmp_job_script, 'w') as script_file:
            script_file.write(job_script)
        
        # Submit the job
        # subprocess.run(["qsub", tmp_job_script])
        print(f"Submitted job for: {json_data['name']}")

if __name__ == "__main__":
    json_files = load_json_files("sal_classification/conditions")  # Adjust path to your JSON files
    submit_jobs(json_files)