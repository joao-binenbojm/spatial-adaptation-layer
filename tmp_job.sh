#!/bin/bash
#PBS -l select=1:ncpus=4:mem=30gb:ngpus=1
#PBS -l walltime=02:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate meta-adabatch
cd spatial-adaptation-layer

PYTHONPATH="$(pwd)" python3 ./sal_classification/intrasession_new.py tmp 
