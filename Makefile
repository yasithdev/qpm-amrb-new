COMMON_ARGS = --gres=gpu:1 --cpus-per-task=1 --mem=64G --time=1-0

WAHAB_ARGS = ${COMMON_ARGS}
FASRC_ARGS = ${COMMON_ARGS} --account=wadduwage_lab

WAHAB_TEST_ARGS = ${WAHAB_ARGS} --partition=timed-gpu
WAHAB_PROD_ARGS = ${WAHAB_ARGS} --partition=gpu
FASRC_TEST_ARGS = ${FASRC_ARGS} --partition=gpu_test
FASRC_PROD_ARGS = ${FASRC_ARGS} --partition=gpu

RUN_ARGS = srun ${WAHAB_PROD_ARGS}

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
srun-jupyter-server:
	${RUN_ARGS} jupyter server --port=8888 --no-browser --ip=0.0.0.0

# ==========
# BATCH JOBS
# ==========

train-mnist-%:
	DATASET_NAME=MNIST MANIFOLD_D=32  MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
train-cifar10-%:
	DATASET_NAME=CIFAR10 MANIFOLD_D=64  MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
train-amrb2_species-%:
	DATASET_NAME=AMRB2_species MANIFOLD_D=512 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
train-amrb2_strain-%:
	DATASET_NAME=AMRB2_strain MANIFOLD_D=512 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
train-qpm_species-%:
	DATASET_NAME=QPM_species MANIFOLD_D=512 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py
train-qpm_species-%:
	DATASET_NAME=QPM_strain MANIFOLD_D=512 MODEL_NAME=$* ${RUN_ARGS} python -u 2_training.py