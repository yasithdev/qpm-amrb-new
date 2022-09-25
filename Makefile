clean:
	rm -rf **/__pycache__


# PREVIEW FUNCTIONS
preview-%:
	@DATASET_NAME=$* python -u 1_amrb_preview.py

# TRAINING FUNCTIONS
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

hpc-train:
	@for MODEL_NAME in "resnet" "capsnet" "efficientcaps" "drcaps"; do \
		for DATASET_NAME in "AMRB_1" "AMRB_2"; do \
			for LABEL_TYPE in "class" "type" "strain" "gram"; do \
				sbatch -J qpm-amrb_train_$${MODEL_NAME}_$${DATASET_NAME}_$${LABEL_TYPE} --export=ALL,SCRIPT_NAME=2_training,MODEL_NAME=$${MODEL_NAME},DATASET_NAME=$${DATASET_NAME},LABEL_TYPE=$${LABEL_TYPE} jobscript.sh; \
			done; \
		done; \
	done

