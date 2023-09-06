# COMMON_ARGS = --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=1-0

# WAHAB_ARGS = ${COMMON_ARGS}
# FASRC_ARGS = ${COMMON_ARGS} --account=wadduwage_lab

# WAHAB_TEST_ARGS = ${WAHAB_ARGS} --partition=timed-gpu
# WAHAB_PROD_ARGS = ${WAHAB_ARGS} --partition=gpu
# FASRC_TEST_ARGS = ${FASRC_ARGS} --partition=gpu_test
# FASRC_PROD_ARGS = ${FASRC_ARGS} --partition=gpu

# RUN_ARGS = srun ${WAHAB_PROD_ARGS}

# ==========
# BATCH JOBS
# ==========

train-mnist-%:
	DATASET_NAME=MNIST EMB_DIMS=32  MODEL_NAME=$* python -u 2_training.py
train-cifar10-%:
	DATASET_NAME=CIFAR10 EMB_DIMS=64  MODEL_NAME=$* python -u 2_training.py
train-amrb2_species-%:
	DATASET_NAME=AMRB2_species EMB_DIMS=512 MODEL_NAME=$* python -u 2_training.py
train-amrb2_strain-%:
	DATASET_NAME=AMRB2_strain EMB_DIMS=512 MODEL_NAME=$* python -u 2_training.py
train-qpm_species-%:
	DATASET_NAME=QPM_species EMB_DIMS=512 MODEL_NAME=$* python -u 2_training.py
train-qpm_strain-%:
	DATASET_NAME=QPM_strain EMB_DIMS=512 MODEL_NAME=$* python -u 2_training.py
