# Train Commands

python -u train.py --dataset_name="mnist" --manifold_d=128 --model_name="flow_ce_mse" --checkpoint_metric="val_decoder_mse" --checkpoint_mode="min"