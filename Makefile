clean:
	rm -rf **/__pycache__

amrb_1:
	for i in $$(seq 0 9); do \
		echo "DS_NAME=AMRB_1 CROSSVAL_K=$${i} python -u 1_amrb_preview.py"; \
		DS_NAME=AMRB_1 CROSSVAL_K=$${i} python -u 2_training.py; \
	done

amrb_2:
	for i in $$(seq 0 9); do \
		DS_NAME=AMRB_2 CROSSVAL_K=$${i} python -u 1_amrb_preview.py; \
	done