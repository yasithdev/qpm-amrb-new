clean:
	python -u wandb_cleanup_runs.py

# dimensionalities used
# * MNIST - 256
# * CIFAR10 - 512
# * QPM_species - 512
# * rbc_phase - 512

# resnet autoencoder (no classifier, mse)
train_resnet_mse_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet_mse_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet_mse_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3";
train_resnet_mse_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2:3";
train_resnet_mse_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0";
train_resnet_mse_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0";

# resnet classifier - ce + mse
train_resnet_ce_mse_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet_ce_mse_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet_ce_mse_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet_ce_mse_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet_ce_mse_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet_ce_mse_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet classifier - edl + mse
train_resnet_edl_mse_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet_edl_mse_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet_edl_mse_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet_edl_mse_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet_edl_mse_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet_edl_mse_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet_edl_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# rescaps classifier - ce + mse
train_rescaps_ce_mse_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_rescaps_ce_mse_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_rescaps_ce_mse_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rescaps_ce_mse_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_rescaps_ce_mse_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="rescaps_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_rescaps_ce_mse_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="rescaps_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 classifier - ce + vicreg
train_resnet18_vicreg_ce_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_vicreg_ce_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_vicreg_ce_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet18_vicreg_ce_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet18_vicreg_ce_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet18_vicreg_ce_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet50 classifier - ce + vicreg
train_resnet50_vicreg_ce_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet50_vicreg_ce_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet50_vicreg_ce_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet50_vicreg_ce_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet50_vicreg_ce_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet50_vicreg_ce_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 SSL model - vicreg
train_resnet18_vicreg_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet18_vicreg_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet18_vicreg_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:2:3";
train_resnet18_vicreg_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="2:3";
train_resnet18_vicreg_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";
train_resnet18_vicreg_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";

# resnet50 SSL model - vicreg
train_resnet50_vicreg_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet50_vicreg_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_resnet50_vicreg_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:2:3";
train_resnet50_vicreg_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="2:3";
train_resnet50_vicreg_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";
train_resnet50_vicreg_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";

# resnet18 hypothesis tester - direct, linear
train_resnet18_ht_enc_mnist:
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_ht_enc_cifar10:
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_ht_enc_qpmb:
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet18_ht_enc_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet18_ht_enc_rbcp:
	python -u train.py --dataset_name="rbc_phase" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet18_ht_enc_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 hypothesis tester - embeddings, linear
train_resnet18_ht_emb_mnist:
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_ht_emb_cifar10:
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_resnet18_ht_emb_qpmb:
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_resnet18_ht_emb_qpmb2:
	python -u train.py --emb_name="QPM2_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM2_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1"; \
	python -u train.py --emb_name="QPM2_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="2:3";
train_resnet18_ht_emb_rbcp:
	python -u train.py --emb_name="rbc_phase_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";
train_resnet18_ht_emb_rbcp2:
	python -u train.py --emb_name="rbc2_phase_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# flow models (reduced dimensionality)
train_flow_ss_vcr_mse_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_flow_ss_vcr_mse_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_flow_ss_vcr_mse_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3";
train_flow_ss_vcr_mse_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2:3";
train_flow_ss_vcr_mse_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1";
train_flow_ss_vcr_mse_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="flow_ss_vcr_mse" --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1";

# cfm models (full dimensionality)
train_unet_otcfm_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="5:6:7:8:9";
train_unet_otcfm_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="5:6:7:8:9";
train_unet_otcfm_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:2:3";
train_unet_otcfm_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="2:3";
train_unet_otcfm_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1";
train_unet_otcfm_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0"; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1";

# cfm models (reduced dimensionality)
train_unet_otcfm_sd_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="5:6:7:8:9";
train_unet_otcfm_cd_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="5:6:7:8:9";
train_unet_otcfm_sd_qpmb:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:2:3";
train_unet_otcfm_sd_qpmb2:
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0:1"; \
	python -u train.py --dataset_name="QPM2_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="2:3";
train_unet_otcfm_sd_rbcp:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1";
train_unet_otcfm_sd_rbcp2:
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood=""; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="0"; \
	python -u train.py --dataset_name="rbc2_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=768 --ckpt_metric="val_loss" --ckpt_mode="min" --train_epochs=500 --patience=500 --ood="1";

# evaluation automation
eval_resnet_mse:
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_mnist.ipynb -p dataset_name "MNIST" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_mnist_A.ipynb -p dataset_name "MNIST" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_mnist_B.ipynb -p dataset_name "MNIST" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_cifar10.ipynb -p dataset_name "CIFAR10" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_cifar10_A.ipynb -p dataset_name "CIFAR10" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_cifar10_B.ipynb -p dataset_name "CIFAR10" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpmb.ipynb -p dataset_name "QPM_species" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpmb_A.ipynb -p dataset_name "QPM_species" -r ood "1:4"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpmb_B.ipynb -p dataset_name "QPM_species" -r ood "0:2:3"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpm2.ipynb -p dataset_name "QPM2_species" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpm2_A.ipynb -p dataset_name "QPM2_species" -r ood "0:1"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_qpm2_B.ipynb -p dataset_name "QPM2_species" -r ood "2:3"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp.ipynb -p dataset_name "rbc_phase" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp_A.ipynb -p dataset_name "rbc_phase" -r ood "1"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp_B.ipynb -p dataset_name "rbc_phase" -r ood "0"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp2.ipynb -p dataset_name "rbc2_phase" -r ood ""; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp2_A.ipynb -p dataset_name "rbc2_phase" -r ood "1"; \
	papermill templates/resnet_mse.ipynb notebooks/eval_resnet_mse_rbcp2_B.ipynb -p dataset_name "rbc2_phase" -r ood "0";
eval_resnet_ce_mse:
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_mnist.ipynb -p dataset_name "MNIST" -r ood ""; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_mnist_A.ipynb -p dataset_name "MNIST" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_mnist_B.ipynb -p dataset_name "MNIST" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_cifar10.ipynb -p dataset_name "CIFAR10" -r ood ""; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_cifar10_A.ipynb -p dataset_name "CIFAR10" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_cifar10_B.ipynb -p dataset_name "CIFAR10" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpmb.ipynb -p dataset_name "QPM_species" -r ood ""; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpmb_A.ipynb -p dataset_name "QPM_species" -r ood "1:4"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpmb_B.ipynb -p dataset_name "QPM_species" -r ood "0:2:3"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpm2.ipynb -p dataset_name "QPM2_species" -r ood ""; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpm2_A.ipynb -p dataset_name "QPM2_species" -r ood "0:1"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_qpm2_B.ipynb -p dataset_name "QPM2_species" -r ood "2:3"; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_rbcp.ipynb -p dataset_name "rbc_phase" -r ood ""; \
	papermill templates/resnet_ce_mse.ipynb notebooks/eval_resnet_ce_mse_rbcp2.ipynb -p dataset_name "rbc2_phase" -r ood "";
eval_resnet_edl_mse:
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_mnist.ipynb -p dataset_name "MNIST" -r ood ""; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_mnist_A.ipynb -p dataset_name "MNIST" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_mnist_B.ipynb -p dataset_name "MNIST" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_cifar10.ipynb -p dataset_name "CIFAR10" -r ood ""; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_cifar10_A.ipynb -p dataset_name "CIFAR10" -r ood "0:1:2:3:4"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_cifar10_B.ipynb -p dataset_name "CIFAR10" -r ood "5:6:7:8:9"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpmb.ipynb -p dataset_name "QPM_species" -r ood ""; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpmb_A.ipynb -p dataset_name "QPM_species" -r ood "1:4"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpmb_B.ipynb -p dataset_name "QPM_species" -r ood "0:2:3"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpm2.ipynb -p dataset_name "QPM2_species" -r ood ""; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpm2_A.ipynb -p dataset_name "QPM2_species" -r ood "0:1"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_qpm2_B.ipynb -p dataset_name "QPM2_species" -r ood "2:3"; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_rbcp.ipynb -p dataset_name "rbc_phase" -r ood ""; \
	papermill templates/resnet_edl_mse.ipynb notebooks/eval_resnet_edl_mse_rbcp2.ipynb -p dataset_name "rbc2_phase" -r ood "";
eval_resnet50_vicreg_ce:
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_mnist.ipynb -p dataset_name "MNIST" -r ood ""; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_mnist_A.ipynb -p dataset_name "MNIST" -r ood "0:1:2:3:4"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_mnist_B.ipynb -p dataset_name "MNIST" -r ood "5:6:7:8:9"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_cifar10.ipynb -p dataset_name "CIFAR10" -r ood ""; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_cifar10_A.ipynb -p dataset_name "CIFAR10" -r ood "0:1:2:3:4"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_cifar10_B.ipynb -p dataset_name "CIFAR10" -r ood "5:6:7:8:9"; \
	papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpmb.ipynb -p dataset_name "QPM_species" -r ood ""; \
	papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpmb_A.ipynb -p dataset_name "QPM_species" -r ood "1:4"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpmb_B.ipynb -p dataset_name "QPM_species" -r ood "0:2:3"; \
	papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpm2.ipynb -p dataset_name "QPM2_species" -r ood ""; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpm2_A.ipynb -p dataset_name "QPM2_species" -r ood "0:1"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_qpm2_B.ipynb -p dataset_name "QPM2_species" -r ood "2:3"; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_rbcp.ipynb -p dataset_name "rbc_phase" -r ood ""; \
	#papermill templates/resnet50_vicreg_ce.ipynb notebooks/eval_resnet50_vicreg_ce_rbcp2.ipynb -p dataset_name "rbc2_phase" -r ood "";
eval_flow_ss_vcr_mse:
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_mnist.ipynb -p dataset_name "MNIST" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_mnist_A.ipynb -p dataset_name "MNIST" -r ood "0:1:2:3:4"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_mnist_B.ipynb -p dataset_name "MNIST" -r ood "5:6:7:8:9"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_cifar10.ipynb -p dataset_name "CIFAR10" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_cifar10_A.ipynb -p dataset_name "CIFAR10" -r ood "0:1:2:3:4"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_cifar10_B.ipynb -p dataset_name "CIFAR10" -r ood "5:6:7:8:9"; \
	papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpmb.ipynb -p dataset_name "QPM_species" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpmb_A.ipynb -p dataset_name "QPM_species" -r ood "1:4"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpmb_B.ipynb -p dataset_name "QPM_species" -r ood "0:2:3"; \
	papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpm2.ipynb -p dataset_name "QPM2_species" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpm2_A.ipynb -p dataset_name "QPM2_species" -r ood "0:1"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_qpm2_B.ipynb -p dataset_name "QPM2_species" -r ood "2:3"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp.ipynb -p dataset_name "rbc_phase" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp_A.ipynb -p dataset_name "rbc_phase" -r ood "1"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp_B.ipynb -p dataset_name "rbc_phase" -r ood "0"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp2.ipynb -p dataset_name "rbc2_phase" -r ood ""; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp2_A.ipynb -p dataset_name "rbc2_phase" -r ood "1"; \
	#papermill templates/flow_ss_vcr_mse.ipynb notebooks/eval_flow_ss_vcr_mse_rbcp2_B.ipynb -p dataset_name "rbc2_phase" -r ood "0";

eval_ae: eval_resnet_ce_mse eval_resnet_edl_mse eval_resnet_mse

eval_bm: eval_resnet50_vicreg_ce

eval_flow: eval_flow_ss_vcr_mse

eval: eval_bm eval_ae eval_flow

dataset_stats:
	python index_labels.py --dataset_name="MNIST"        --ood=""; \
	python index_labels.py --dataset_name="MNIST"        --ood="0:1:2:3:4"; \
	python index_labels.py --dataset_name="MNIST"        --ood="5:6:7:8:9"; \
	python index_labels.py --dataset_name="CIFAR10"      --ood=""; \
	python index_labels.py --dataset_name="CIFAR10"      --ood="0:1:2:3:4"; \
	python index_labels.py --dataset_name="CIFAR10"      --ood="5:6:7:8:9"; \
	python index_labels.py --dataset_name="QPM_species"  --ood=""; \
	python index_labels.py --dataset_name="QPM_species"  --ood="1:4"; \
	python index_labels.py --dataset_name="QPM_species"  --ood="0:2:3"; \
	python index_labels.py --dataset_name="QPM2_species" --ood=""; \
	python index_labels.py --dataset_name="QPM2_species" --ood="0:1"; \
	python index_labels.py --dataset_name="QPM2_species" --ood="2:3"; \
	python index_labels.py --dataset_name="rbc_phase"    --ood=""; \
	python index_labels.py --dataset_name="rbc_phase"    --ood="1"; \
	python index_labels.py --dataset_name="rbc_phase"    --ood="0"; \
	python index_labels.py --dataset_name="rbc2_phase"    --ood=""; \
	python index_labels.py --dataset_name="rbc2_phase"    --ood="1"; \
	python index_labels.py --dataset_name="rbc2_phase"    --ood="0";
