clean:
	rm -rf **/__pycache__

amrb_1_preview:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_1 CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	done

amrb_2_preview:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_2 CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	done

amrb_1_train-%:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_1 MODEL_NAME=$* CROSSVAL_K=$${i} python -u 2_training.py; \
	done

amrb_2_train-%:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_2 MODEL_NAME=$* CROSSVAL_K=$${i} python -u 2_training.py; \
	done

hpc_preview:
	for ds in "AMRB_1" "AMRB_2"; do \
		DATASET_NAME=$${ds} SCRIPT_NAME="1_amrb_preview.py" sbatch -J preview-$${ds} jobscript.sh; \
	done

hpc_train:
	for ds in "AMRB_1" "AMRB_2"; do \
		for model in "resnet" "capsnet" "deepcaps" "efficientcaps"; do \
			DATASET_NAME=$${ds} MODEL_NAME=$${model} SCRIPT_NAME="2_training.py" sbatch -J train-$${model}-$${ds} jobscript.sh; \
		done; \
	done

