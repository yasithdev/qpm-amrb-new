clean:
	rm -rf **/__pycache__

amrb_1:
	DS_NAME=AMRB_1 python -u 1_amrb_preview.py

amrb_2:
	DS_NAME=AMRB_2 python -u 1_amrb_preview.py