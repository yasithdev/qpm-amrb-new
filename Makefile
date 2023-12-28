wandb_fetch:
	python -u wandb_cleanup_runs.py; \
	python -u wandb_get_runs.py; \


# classification models

train_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \

train_mnist_A:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
    
train_mnist_B:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \

train_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \

train_cifar10_A:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
    
train_cifar10_B:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:1:2:3:4"; \

train_qpm_species:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \

train_qpm_species_A:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \

train_qpm_species_B:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \

train_rbc_phase:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
    
train_rbc_phase_A:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
    
train_rbc_phase_B:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \


# SSL models

train_qpm_species_resnet50_simclr:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="1:4"; \
    python -u train.py --dataset_name="QPM_species" --model_name="resnet50_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="0:2:3"; \

train_qpm_species_resnet50_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="0:2:3"; \

train_qpm_species_resnet18_simclr:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="0:2:3"; \

train_qpm_species_resnet18_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="0:2:3"; \


# permutation models (from SSL embeddings)

train_qpm_species_emb_resnet50_vicreg_ht_linear:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \
    
train_qpm_species_emb_resnet50_simclr_ht_linear:
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \

train_qpm_species_emb_resnet50_vicreg_ht_mlp:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \


train_qpm_species_emb_resnet50_simclr_ht_mlp:
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \


# permutation models (direct)

train_qpm_species_resnet50_ht_linear:
	python -u train.py --dataset_name="QPM_species" --model_name="ht_linear_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="ht_linear_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="ht_linear_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \
    
train_qpm_species_resnet50_ht_mlp:
	python -u train.py --dataset_name="QPM_species" --model_name="ht_mlp_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="ht_mlp_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="ht_mlp_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="0:2:3"; \


# flow models (reduced dimensionality)

train_mnist_flow_ss_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_mnist_flow_ms_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_mnist_flow_vcr_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_ss_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_ms_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_vcr_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_qpm_species_flow_ss_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3"; \

train_qpm_species_flow_ms_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3"; \
    
train_qpm_species_flow_vcr_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3"; \

train_rbc_phase_flow_ss_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \

train_rbc_phase_flow_ms_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \

train_rbc_phase_flow_vcr_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1" \


# conditional flow matching models (full dimensionality)

train_mnist_cfm_otcfm:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_cfm_otcfm:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3072 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_qpm_species_cfm_otcfm:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=3600 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3"; \

train_rbc_phase_cfm_otcfm:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm" --batch_size=128 --emb_dims=4096 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \


# conditional flow matching models (reduced dimensionality via spatial / channel drop)

train_mnist_cfm_otcfm_sd:
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_cfm_otcfm_cd:
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="cfm_otcfm_c" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9; \
    
train_qpm_species_cfm_otcfm_sd:
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1:4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:2:3"; \
    
train_rbc_phase_cfm_otcfm_sd:
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="cfm_otcfm_s" --batch_size=128 --emb_dims=1024 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \


# evaluation automation

eval_qpm_species_flow_ms_mse:
	papermill eval_qpm_species_flow_ms_mse_M128.ipynb assets/runs/eval_qpm_species_flow_ms_mse_M128_ood_0.ipynb -p ood 0; \
	papermill eval_qpm_species_flow_ms_mse_M128.ipynb assets/runs/eval_qpm_species_flow_ms_mse_M128_ood_1.ipynb -p ood 1; \
	papermill eval_qpm_species_flow_ms_mse_M128.ipynb assets/runs/eval_qpm_species_flow_ms_mse_M128_ood_2.ipynb -p ood 2; \
	papermill eval_qpm_species_flow_ms_mse_M128.ipynb assets/runs/eval_qpm_species_flow_ms_mse_M128_ood_3.ipynb -p ood 3; \
	papermill eval_qpm_species_flow_ms_mse_M128.ipynb assets/runs/eval_qpm_species_flow_ms_mse_M128_ood_4.ipynb -p ood 4; \

eval_qpm_species_flow_vcr_mse:
	papermill eval_qpm_species_flow_vcr_mse_M128.ipynb assets/runs/eval_qpm_species_flow_vcr_mse_M128_ood_0.ipynb -p ood 0; \
	#papermill eval_qpm_species_flow_vcr_mse_M128.ipynb assets/runs/eval_qpm_species_flow_vcr_mse_M128_ood_1.ipynb -p ood 1; \
	#papermill eval_qpm_species_flow_vcr_mse_M128.ipynb assets/runs/eval_qpm_species_flow_vcr_mse_M128_ood_2.ipynb -p ood 2; \
	#papermill eval_qpm_species_flow_vcr_mse_M128.ipynb assets/runs/eval_qpm_species_flow_vcr_mse_M128_ood_3.ipynb -p ood 3; \
	#papermill eval_qpm_species_flow_vcr_mse_M128.ipynb assets/runs/eval_qpm_species_flow_vcr_mse_M128_ood_4.ipynb -p ood 4; \

eval_rbc_phase_flow_vcr_mse:
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128_ood_0.ipynb -p ood ""; \
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128_ood_1.ipynb -p ood "0"; \
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128.ipynb -p ood "1" \
    
eval_mnist_flow_vcr_mse:
	papermill eval_mnist_flow_vcr_mse_M128.ipynb assets/runs/eval_mnist_flow_vcr_mse_M128_ood_01234.ipynb -p ood "0:1:2:3:4"; \
	papermill eval_mnist_flow_vcr_mse_M128.ipynb assets/runs/eval_mnist_flow_vcr_mse_M128_ood_56789.ipynb -p ood "5:6:7:8:9"; \
