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
train-resnet-%:
	@DATASET_NAME=$* MODEL_NAME=resnet python -u 2_training.py

train-capsnet-%:
	@DATASET_NAME=$* MODEL_NAME=capsnet python -u 2_training.py

train-efficientcaps-%:
	@DATASET_NAME=$* MODEL_NAME=efficientcaps python -u 2_training.py

train-drcaps-%:
	@DATASET_NAME=$* MODEL_NAME=drcaps python -u 2_training.py

hpc-preview:
	@for DATASET_NAME in "AMRB_1" "AMRB_2"; do \
	for LABEL_TYPE in "class" "type" "strain" "gram"; do \
		sbatch -J qpm-amrb_preview_$${DATASET_NAME}_$${LABEL_TYPE} --export=ALL,SCRIPT_NAME=1_amrb_preview,DATASET_NAME=$${DATASET_NAME},LABEL_TYPE=$${LABEL_TYPE} jobscript.sh; \
	done; \
	done

# ------------------
# AMRB_1 - K-FOLD
# ------------------
hpc-train-amrb1-kfold-class:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_class_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=class,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-type:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_type_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=type,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-strain:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_strain_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=strain,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-kfold-gram:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_k-fold_gram_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=gram,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_2 - K-FOLD
# ------------------
hpc-train-amrb2-kfold-class:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_class_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=class,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-type:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_type_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=type,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-strain:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_strain_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=strain,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-kfold-gram:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_k-fold_gram_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,LABEL_TYPE=gram,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_1 - LEAVE-OUT
# ------------------
hpc-train-amrb1-leaveout-class:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_leave-out_class_$${MODEL_NAME} --array=0-6 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=7,LABEL_TYPE=class,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb1-leaveout-type:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb1_leave-out_type_$${MODEL_NAME} --array=0-4 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=5,LABEL_TYPE=type,DATASET_NAME=AMRB_1,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

# ------------------
# AMRB_2 - LEAVE-OUT
# ------------------
hpc-train-amrb2-leaveout-class:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_leave-out_class_$${MODEL_NAME} --array=0-20 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=21,LABEL_TYPE=class,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-amrb2-leaveout-type:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-amrb2_leave-out_type_$${MODEL_NAME} --array=0-4 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,CV_FOLDS=5,LABEL_TYPE=type,DATASET_NAME=AMRB_2,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done


# ------------------
# MNIST
# ------------------
hpc-train-mnist-kfold:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-mnist_k-fold_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=k-fold,DATASET_NAME=MNIST,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

hpc-train-mnist-leaveout:
	for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		sbatch -J qpm-mnist_leave-out_$${MODEL_NAME} --array=0-9 \
		--export=ALL,SCRIPT_NAME=2_training,CV_MODE=leave-out,DATASET_NAME=MNIST,MODEL_NAME=$${MODEL_NAME} jobscript.sh; \
	done

