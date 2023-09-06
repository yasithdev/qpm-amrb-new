# Train Commands

python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"
python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"