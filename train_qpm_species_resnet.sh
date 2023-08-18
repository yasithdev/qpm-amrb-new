#!/usr/bin/env sh

export DATA_DIR=$HOME/datasets
export EXPERIMENT_BASE=$HOME/experiments/ood_flows
export LOG_LEVEL=INFO
export BATCH_SIZE=64
export OPTIM_LR=0.001
export OPTIM_M=0.8
export TRAIN_EPOCHS=100
export EXC_RESUME=1
export DATASET_NAME=QPM_species
export MANIFOLD_D=512
export CHECKPOINT_METRIC=val_accuracy

# species: Ab, Bs, Ec, Kp, Bs
# choose the ood variant (optional)
# export OOD_K=0                            # leave out Ab
# export OOD_K=2:3                          # leave out Ec:Kp

# export MODEL_NAME=resnet_ce_mse           # crossentropy + mse
export MODEL_NAME=resnet_ce               # just crossentropy
# export MODEL_NAME=resnet_edl_mse          # evidential + mse
# export MODEL_NAME=resnet_edl              # just evidential

python -u train.py
