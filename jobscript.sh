#!/bin/bash

#SBATCH --output=logs/%x-%a.log
#SBATCH --partition=gpu
#SBATCH --time=1-0
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# project name
export PROJECT_NAME=qpm-amrb-new

# activate virtual environment (should already be created)
source ~/.virtualenvs/$PROJECT_NAME/bin/activate

# define project variables
export DATASET_NAME=$DATASET_NAME.$SLURM_ARRAY_TASK_ID

# cd to project directory
cd ~/projects/$PROJECT_NAME

srun --unbuffered python -u $SCRIPT_NAME.py

