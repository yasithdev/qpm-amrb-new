clean:
	rm -rf **/__pycache__

amrb_1:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_1 MODEL_NAME=resnet CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	DATASET_NAME=AMRB_1 MODEL_NAME=resnet CROSSVAL_K=$${i} python -u 2_training.py; \
	done

amrb_2:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_2 MODEL_NAME=resnet CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	DATASET_NAME=AMRB_2 MODEL_NAME=resnet CROSSVAL_K=$${i} python -u 2_training.py; \
	done