clean:
	rm -rf **/__pycache__

amrb_1_preview:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_1 CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	done

amrb_1_train:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_1 MODEL_NAME=capsnet CROSSVAL_K=$${i} python -u 2_training.py; \
	done

amrb_2_preview:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_2 CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	done

amrb_2_train:
	for i in $$(seq 0 9); do \
	DATASET_NAME=AMRB_2 MODEL_NAME=capsnet CROSSVAL_K=$${i} python -u 2_training.py; \
	done