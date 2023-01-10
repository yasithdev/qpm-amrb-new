clean:
	rm -rf **/__pycache__


# ------------------
# PREVIEW FUNCTIONS
# ------------------
preview-%:
	@DATASET_NAME=$* python -u 1_amrb_preview.py

# ------------------
# TRAINING FUNCTIONS
# ------------------
train-flow-%:
	@DATASET_NAME=$* MODEL_NAME=flow python -u 2_training.py

train-resnet-%:
	@DATASET_NAME=$* MODEL_NAME=resnet python -u 2_training.py

train-drcaps-%:
	@DATASET_NAME=$* MODEL_NAME=drcaps python -u 2_training.py

hpc-preview:
	@for DATASET_NAME in "AMRB_1" "AMRB_2"; do \
	for LABEL_TYPE in "strain" "species" "type" "gram"; do \
		sbatch -J qpm-amrb_preview_$${DATASET_NAME}_$${LABEL_TYPE} --export=ALL,SCRIPT_NAME=1_amrb_preview,DATASET_NAME=$${DATASET_NAME},LABEL_TYPE=$${LABEL_TYPE} jobscript.sh; \
	done; \
	done

# ------------------
# AMRB_1 - K-FOLD
# ------------------
hpc-train-amrb1-kfold-strain:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_strain_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=strain,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-species:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_species_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=species,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-type:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_type_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=type,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-gram:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_gram_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=gram,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_2 - K-FOLD
# ------------------
hpc-train-amrb2-kfold-strain:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_strain_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=strain,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-species:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_species_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=species,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-type:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_type_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=type,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-gram:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_gram_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=gram,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_1 - LEAVE-OUT
# ------------------
hpc-train-amrb1-leaveout-strain:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_leave-out_strain_$${MODEL_NAME} --array=0-4 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=5,LABEL_TYPE=strain,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-leaveout-species:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb1_leave-out_species_$${MODEL_NAME} --array=0-2 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=3,LABEL_TYPE=species,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_2 - LEAVE-OUT
# ------------------
hpc-train-amrb2-leaveout-strain:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_leave-out_strain_$${MODEL_NAME} --array=0-18 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=19,LABEL_TYPE=strain,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-leaveout-species:
	for MODEL_NAME in "resnet" "drcaps"; do \
		sbatch -J qpm-amrb2_leave-out_species_$${MODEL_NAME} --array=0-3 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=4,LABEL_TYPE=species,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done


# ------------------
# FLOW
# ------------------
hpc-train-flow-mnist:
	for CV_MODE in "k-fold" "leave-out"; do \
		sbatch -J qpm-mnist_$${CV_MODE}_flow --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=$${CV_MODE},DATASET_NAME=MNIST,MODEL_NAME=flow jobscript.sh; \
	done

hpc-train-flow-cifar10:
	for CV_MODE in "k-fold" "leave-out"; do \
		sbatch -J qpm-cifar10_$${CV_MODE}_flow --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=$${CV_MODE},DATASET_NAME=CIFAR10,MODEL_NAME=flow jobscript.sh; \
	done

# ------------------
# Interactive Jobs
# ------------------

srun-gpu-test-wahab:
	srun --partition=timed-gpu --gres=gpu:1 --cpus-per-task=1 --pty bash

srun-gpu-wahab:
	srun --partition=gpu --gres=gpu:1 --cpus-per-task=1 --pty bash

srun-gpu-test-fasrc:
	srun --account=wadduwage_lab --partition=gpu_test --time=0-6 --gres=gpu:1 --mem=64G --pty bash

srun-gpu-fasrc:
	srun --account=wadduwage_lab --partition=gpu --time=0-6 --gres=gpu:1 --mem=64G --pty bash