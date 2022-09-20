#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --output=~/logs/%j-%a.log
#SBATCH --array=0-9

# load required modules
module load cuda/10.1.186
module load cudnn/10.1.7.5

# project name
export PROJECT_NAME=qpm-amrb-new

# activate virtual environment (should already be created)
source ~/.virtualenvs/$PROJECT_NAME/bin/activate

# define project variables
export DATA_DIR=~/datasets
export EXPERIMENT_DIR=~/experiments/$PROJECT_NAME
export LOG_DIR=~/logs/$PROJECT_NAME
export CROSSVAL_K=$SLURM_ARRAY_TASK_ID

# cd to project directory
cd ~/projects/$PROJECT_NAME

# don't forget to include the following!
#   -J <job_name>
#   DATASET_NAME=<dataset_name>
#   MODEL_NAME=<model_name>
#   SCRIPT_NAME=<script_name>

python -u $SCRIPT_NAME

