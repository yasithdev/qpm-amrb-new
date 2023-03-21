COMMON_ARGS = --gres=gpu:1 --cpus-per-task=1 --mem=64G --time=1-0

WAHAB_ARGS = ${COMMON_ARGS}
FASRC_ARGS = ${COMMON_ARGS} --account=wadduwage_lab

WAHAB_TEST_ARGS = ${WAHAB_ARGS} --partition=timed-gpu
WAHAB_PROD_ARGS = ${WAHAB_ARGS} --partition=gpu
FASRC_TEST_ARGS = ${FASRC_ARGS} --partition=gpu_test
FASRC_PROD_ARGS = ${FASRC_ARGS} --partition=gpu

# ================
# INTERACTIVE JOBS
# ================

srun-wahab-test:
	srun ${WAHAB_TEST_ARGS} --pty bash
srun-wahab:
	srun ${WAHAB_PROD_ARGS} --pty bash
srun-fasrc-test:
	srun ${FASRC_TEST_ARGS} --pty bash
srun-fasrc:
	srun ${FASRC_PROD_ARGS} --pty bash

# ==========
# BATCH JOBS
# ==========

RUN_ARGS = srun ${WAHAB_PROD_ARGS}

# == [BENCHMARK] UMAP (X) ==
umapx-mnist-kfold:
	DATASET_NAME=MNIST.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-mnist-leaveout:
	DATASET_NAME=MNIST.$${CV_K}   CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-cifar10-kfold:
	DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-cifar10-leaveout:
	DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py

# == [BENCHMARK] TRAIN BASE MODELS ==
train-mnist-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=MNIST.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-mnist-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=MNIST.$${CV_K}   CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-cifar10-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-cifar10-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py

# == [BENCHMARK] UMAP BASE MODELS (Z) ==
umapz-mnist-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=MNIST.$${CV_K}  CV_MODE=k-fold     CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-mnist-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=MNIST.$${CV_K}  CV_MODE=leave-out  CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-cifar10-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-cifar10-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py

# == [BENCHMARK] TRAIN HYBRID MODELS ==
train-mnist-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=MNIST.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-mnist-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=MNIST.$${CV_K}   CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-cifar10-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-cifar10-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py

# == [BENCHMARK] UMAP HYBRID MODELS (Z) ==
umapz-mnist-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=MNIST.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-mnist-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=MNIST.$${CV_K}   CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-cifar10-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-cifar10-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=CIFAR10.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py


# == [AMRB2] UMAP (X) ==
umapx-amrb2-strain-kfold:
	DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-amrb2-strain-leaveout:
	DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=leave-out CV_FOLDS=19 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-amrb2-species-kfold:
	DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 1_amrb_preview.py
umapx-amrb2-species-leaveout:
	DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=leave-out CV_FOLDS=4  ${RUN_ARGS} python -u 1_amrb_preview.py

# == [AMRB2] TRAIN BASE MODELS ==
train-amrb2-strain-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-amrb2-strain-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=leave-out CV_FOLDS=19 ${RUN_ARGS} python -u 2_training.py
train-amrb2-species-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-amrb2-species-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=leave-out CV_FOLDS=4  ${RUN_ARGS} python -u 2_training.py

# == [AMRB2] UMAP BASE MODELS (Z) ==
umapz-amrb2-strain-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-strain-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_strain.$${CV_K}  CV_MODE=leave-out CV_FOLDS=19 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-species-kfold-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-species-leaveout-base-%:
	MODEL_NAME=$* DATASET_NAME=AMRB2_species.$${CV_K} CV_MODE=leave-out CV_FOLDS=4  ${RUN_ARGS} python -u 5_umap.py

# == [AMRB2] TRAIN HYBRID MODELS ==
train-amrb2-strain-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=AMRB2_strain.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-amrb2-strain-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=AMRB2_strain.$${CV_K}   CV_MODE=leave-out CV_FOLDS=19 ${RUN_ARGS} python -u 2_training.py
train-amrb2-species-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=AMRB2_species.$${CV_K}  CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 2_training.py
train-amrb2-species-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=AMRB2_species.$${CV_K}  CV_MODE=leave-out CV_FOLDS=4  ${RUN_ARGS} python -u 2_training.py

# == [AMRB2] UMAP HYBRID MODELS (Z) ==
umapz-amrb2-strain-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=amrb2_strain.$${CV_K}   CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-strain-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=amrb2_strain.$${CV_K}   CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-species-kfold-hybrid:
	MODEL_NAME=flow DATASET_NAME=amrb2_species.$${CV_K} CV_MODE=k-fold    CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
umapz-amrb2-species-leaveout-hybrid:
	MODEL_NAME=flow DATASET_NAME=amrb2_species.$${CV_K} CV_MODE=leave-out CV_FOLDS=10 ${RUN_ARGS} python -u 5_umap.py
