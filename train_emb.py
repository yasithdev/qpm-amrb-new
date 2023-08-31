import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from config import load_config

# initialize the RNG deterministically
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision("medium")

config = load_config()

emb_path = f"assets/embeddings/{config.dataset_name}.npz"
config.load_embedding(embedding_path=emb_path, num_dims=config.manifold_d, num_targets=21)

model = config.get_model(model_name=config.model_name, in_dims=config.manifold_d, rand_dims=1000)

run_name = f"{config.dataset_name}_{config.model_name}"
wandb_logger = WandbLogger(
    project="uq_project", log_model="all", name=run_name, config={**config.as_dict(), **model.hparams}
)
wandb_logger.watch(model, log="all")

checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=config.train_epochs,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
)

trainer.fit(model=model, datamodule=config.datamodule)
