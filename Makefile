wandb_fetch:
	python -u wandb_cleanup_runs.py; \
	python -u wandb_get_runs.py;

# dimensionalities used
# * MNIST - 256
# * CIFAR10 - 512
# * QPM_species - 512
# * rbc_phase - 512

# resnet classifier - ce + mse
train_mnist_resnet_ce_mse:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet_ce_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet_ce_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet_ce_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet classifier - edl + mse
train_mnist_resnet_edl_mse:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet_edl_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet_edl_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet_edl_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# rescaps classifier - ce + mse
train_mnist_rescaps_ce_mse:
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="rescaps_ce_mse" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_rescaps_ce_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_rescaps_ce_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="rescaps_ce_mse" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_rescaps_ce_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="rescaps_ce_mse" --emb_dims=512 --batch_size=16 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 classifier - ce + vicreg
train_mnist_resnet18_vicreg_ce:
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet18_vicreg_ce:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet18_vicreg_ce:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet18_vicreg_ce:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet18_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet50 classifier - ce + vicreg
train_mnist_resnet50_vicreg_ce:
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg_ce" --emb_dims=256 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet50_vicreg_ce:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet50_vicreg_ce:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet50_vicreg_ce:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet50_vicreg_ce" --emb_dims=512 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 SSL model - vicreg
train_mnist_resnet18_vicreg:
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_cifar10_resnet18_vicreg:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_qpm_species_resnet18_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:2:3";
train_rbc_phase_resnet18_vicreg:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet18_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";

# resnet50 SSL model - vicreg
train_mnist_resnet50_vicreg:
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_cifar10_resnet50_vicreg:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:1:2:3:4";
train_qpm_species_resnet50_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="0:2:3";
train_rbc_phase_resnet50_vicreg:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet50_vicreg" --emb_dims=512 --ckpt_metric="val_loss_emb" --ckpt_mode="min" --ood="";

# resnet18 hypothesis tester - direct, linear
train_mnist_resnet18_ht_linear:
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet18_ht_linear:
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet18_ht_linear:
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet18_ht_linear:
	python -u train.py --dataset_name="rbc_phase" --emb_dims=512 --rand_perms=500 --model_name="ht_linear_enc_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# resnet18 hypothesis tester - embeddings, linear
train_mnist_resnet18_vicreg_emb_ht_linear:
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --emb_name="MNIST_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_cifar10_resnet18_vicreg_emb_ht_linear:
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --emb_name="CIFAR10_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4";
train_qpm_species_resnet18_vicreg_emb_ht_linear:
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3";
train_rbc_phase_resnet18_vicreg_emb_ht_linear:
	python -u train.py --emb_name="rbc_phase_resnet18_vicreg" --emb_dims=512 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="";

# flow models (reduced dimensionality)
train_mnist_flow_ss_vcr_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_cifar10_flow_ss_vcr_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_qpm_species_flow_ss_vcr_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3";
train_rbc_phase_flow_ss_vcr_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_vcr_mse" --batch_size=32 --emb_dims=512 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1";

# cfm models (full dimensionality)
train_mnist_unet_otcfm:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_cifar10_unet_otcfm:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_qpm_species_unet_otcfm:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3";
train_rbc_phase_unet_otcfm:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=256 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1";

# cfm models (reduced dimensionality)
train_mnist_unet_otcfm_sd:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9";
train_cifar10_unet_otcfm_cd:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9;
train_qpm_species_unet_otcfm_sd:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3";
train_rbc_phase_unet_otcfm_sd:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=256 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1";

# evaluation automation
eval_qpm_species_flow_ss_vcr_mse:
	papermill eval_qpm_species_flow_ss_vcr_mse.ipynb assets/runs/eval_qpm_species_flow_ss_vcr_mse.ipynb -p ood ""; \
	papermill eval_qpm_species_flow_ss_vcr_mse.ipynb assets/runs/eval_qpm_species_flow_ss_vcr_mse_A.ipynb -p ood "1:4"; \
	papermill eval_qpm_species_flow_ss_vcr_mse.ipynb assets/runs/eval_qpm_species_flow_ss_vcr_mse_B.ipynb -p ood "0:2:3";
eval_rbc_phase_flow_ss_vcr_mse:
	papermill eval_rbc_phase_flow_ss_vcr_mse.ipynb assets/runs/eval_rbc_phase_flow_ss_vcr_mse.ipynb -p ood ""; \
	papermill eval_rbc_phase_flow_ss_vcr_mse.ipynb assets/runs/eval_rbc_phase_flow_ss_vcr_mse_A.ipynb -p ood "0"; \
	papermill eval_rbc_phase_flow_ss_vcr_mse.ipynb assets/runs/eval_rbc_phase_flow_ss_vcr_mse_B.ipynb -p ood "1";
eval_mnist_flow_ss_vcr_mse:
	papermill eval_mnist_flow_ss_vcr_mse.ipynb assets/runs/eval_mnist_flow_ss_vcr_mse.ipynb -p ood ""; \
	papermill eval_mnist_flow_ss_vcr_mse.ipynb assets/runs/eval_mnist_flow_ss_vcr_mse_A.ipynb -p ood "0:1:2:3:4"; \
	papermill eval_mnist_flow_ss_vcr_mse.ipynb assets/runs/eval_mnist_flow_ss_vcr_mse_B.ipynb -p ood "5:6:7:8:9";
