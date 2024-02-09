#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --partition=gpu_requeue
#SBATCH --account=wadduwage_lab
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@90

# activate conda env
conda activate qpm-amrb

# run script from above
export HOME=/n/home12/yasith
srun python -u train.py --dataset_name="QPM2_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2:3";
