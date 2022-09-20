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

train-deepcaps-%:
	@DATASET_NAME=$* MODEL_NAME=deepcaps python -u 2_training.py

train-efficientcaps-%:
	@DATASET_NAME=$* MODEL_NAME=efficientcaps python -u 2_training.py

hpc-preview:
	@for DATASET_NAME in "AMRB_1" "AMRB_2"; do \
		echo "sbatch -J qpm-amrb-new/preview/$${DATASET_NAME} --export=ALL,SCRIPT_NAME=1_amrb_preview jobscript.sh"; \
	done

hpc-train:
	@for MODEL_NAME in "resnet" "capsnet" "deepcaps" "efficientcaps"; do \
		for DATASET_NAME in "AMRB_1" "AMRB_2"; do \
			echo "sbatch -J qpm-amrb-new/train-$${MODEL_NAME}/$${DATASET_NAME} --export=ALL,SCRIPT_NAME=2_training jobscript.sh"; \
		done; \
	done

