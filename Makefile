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

# == [BENCHMARK] MNIST kfold ==
umapx-mnist-kfold:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=k-fold    CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-mnist-kfold-%:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-mnist-kfold-%:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [BENCHMARK] MNIST leaveout ==
umapx-mnist-leaveout:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=leave-out CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-mnist-leaveout-%:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=leave-out CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
plot-mnist-leaveout-%:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=leave-out CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 4_dnf_plot.py
umapz-mnist-leaveout-%:
	DATASET_NAME=MNIST.$${CV_K} MANIFOLD_D=32 CV_MODE=leave-out CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [BENCHMARK] CIFAR10 kfold ==
umapx-cifar10-kfold:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=k-fold    CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-cifar10-kfold-%:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-cifar10-kfold-%:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [BENCHMARK] CIFAR10 leaveout ==
umapx-cifar10-leaveout:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=leave-out CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-cifar10-leaveout-%:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=leave-out CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-cifar10-leaveout-%:
	DATASET_NAME=CIFAR10.$${CV_K} MANIFOLD_D=64 CV_MODE=leave-out CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [ACTUAL] AMRB2_species kfold ==
umapx-amrb2_species-kfold:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-amrb2_species-kfold-%:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-amrb2_species-kfold-%:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [ACTUAL] AMRB2_species leaveout ==
umapx-amrb2_species-leaveout:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=4                ${RUN_ARGS} python -u 1_umap_x.py
train-amrb2_species-leaveout-%:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=4  MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-amrb2_species-leaveout-%:
	DATASET_NAME=AMRB2_species.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=4  MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [ACTUAL] AMRB2_strain kfold ==
umapx-amrb2_strain-kfold:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10               ${RUN_ARGS} python -u 1_umap_x.py
train-amrb2_strain-kfold-%:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-amrb2_strain-kfold-%:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=k-fold    CV_FOLDS=10 MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py

# == [ACTUAL] AMRB2_strain leaveout ==
umapx-amrb2_strain-leaveout:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=19                ${RUN_ARGS} python -u 1_umap_x.py
train-amrb2_strain-leaveout-%:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=19  MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
umapz-amrb2_strain-leaveout-%:
	DATASET_NAME=AMRB2_strain.$${CV_K} MANIFOLD_D=50 CV_MODE=leave-out CV_FOLDS=19  MODEL_NAME=$* ${RUN_ARGS} python -u 3_umap_z.py
