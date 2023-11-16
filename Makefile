wandb_fetch:
	python -u wandb_cleanup_runs.py; \
	python -u wandb_get_runs.py; \

# classification models

train_mnist:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \

train_mnist_ood:
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="MNIST" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \

train_cifar10:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \

train_cifar10_ood:
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="5:6:7:8:9"; \

train_qpm_species:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \

train_qpm_species_ood:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl_mse" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_ce" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet_edl" --emb_dims=128 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:2"; \

train_RBC_phase:
	python -u train.py --dataset_name="RBC_phase" --model_name="resnet_ce_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="RBC_phase" --model_name="resnet_edl_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="RBC_phase" --model_name="resnet_ce" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="RBC_phase" --model_name="resnet_edl" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \

train_rbc_phase:
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_ce" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="resnet_edl" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \

# self supervised learning models

train_qpm_species_resnet50_simclr:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="1:2"; \

train_qpm_species_resnet50_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet50_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="1:2"; \

train_qpm_species_resnet18_simclr:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_simclr" --emb_dims=128 --ckpt_metric="val_loss_simclr" --ckpt_mode="min" --ood="1:2"; \

train_qpm_species_resnet18_vicreg:
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min"; \
	python -u train.py --dataset_name="QPM_species" --model_name="resnet18_vicreg" --emb_dims=128 --ckpt_metric="val_loss_vicreg" --ckpt_mode="min" --ood="1:2"; \

# permutation models

train_qpm_species_emb_resnet50_ht_linear:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_hts=100  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=250  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=1000 --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=100  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=250  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=1000 --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \

train_qpm_species_emb_resnet50_ht_linear_ood:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=100  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=250  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=1000 --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=100  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=250  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500  --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=1000 --model_name="ht_linear_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \

train_qpm_species_emb_resnet50_ht_mlp:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=100 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=250 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=100 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=250 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max"; \

train_qpm_species_emb_resnet50_ht_mlp_ood:
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=100 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=250 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_vicreg_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=100 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=250 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \
	python -u train.py --emb_name="QPM_species_resnet50_simclr_M128" --emb_dims=1024 --rand_perms=500 --model_name="ht_mlp_ce" --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \

train_qpm_species_resnet50_ht_linear:
	python -u train.py --dataset_name="QPM_species" --model_name="ht_linear_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max"; \
	python -u train.py --dataset_name="QPM_species" --model_name="ht_linear_enc_ce" --emb_dims=128 --rand_perms=500 --ckpt_metric="val_accuracy" --ckpt_mode="max" --ood="1:4"; \

# flow models

train_mnist_flow_ss_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_mnist_flow_ms_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_mnist_flow_vcr_mse:
	python -u train.py --dataset_name="MNIST" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="MNIST" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_ss_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_ms_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_cifar10_flow_vcr_mse:
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0:1:2:3:4"; \
	python -u train.py --dataset_name="CIFAR10" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=256 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="5:6:7:8:9"; \

train_qpm_species_flow_ss_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="4"; \

train_qpm_species_flow_ms_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="4"; \
    
train_qpm_species_flow_vcr_mse:
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="4"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="3"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="2"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="QPM_species" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood=""; \

train_rbc_phase_flow_ss_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ss_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \

train_rbc_phase_flow_ms_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_ms_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \

train_rbc_phase_flow_vcr_mse:
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="0"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min" --ood="1"; \
	python -u train.py --dataset_name="rbc_phase" --model_name="flow_vcr_mse" --batch_size=32 --emb_dims=128 --ckpt_metric="val_loss" --ckpt_mode="min"; \

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
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128_ood_0.ipynb -p ood "0"; \
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128_ood_1.ipynb -p ood "1"; \
	papermill eval_rbc_phase_flow_vcr_mse_M128.ipynb assets/runs/eval_rbc_phase_flow_vcr_mse_M128.ipynb -p ood ""; \
    
eval_mnist_flow_vcr_mse:
	papermill eval_mnist_flow_vcr_mse_M128.ipynb assets/runs/eval_mnist_flow_vcr_mse_M128_ood_01234.ipynb -p ood "0:1:2:3:4"; \
	papermill eval_mnist_flow_vcr_mse_M128.ipynb assets/runs/eval_mnist_flow_vcr_mse_M128_ood_56789.ipynb -p ood "5:6:7:8:9"; \
