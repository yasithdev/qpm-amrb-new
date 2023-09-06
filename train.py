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

# initialize config, data attributes, and loaders
config = load_config()
config.load_data()
config.print_labels()
assert config.datamodule

model = config.get_model()
wandb_logger = WandbLogger(
    project="robust_ml",
    log_model="all",
    name=f"{config.dataset_name}_{config.model_name}",
    config={**config.params, **model.hparams},
)
wandb_logger.watch(model, log="all")

checkpoint_callback = ModelCheckpoint(monitor=config.ckpt_metric, mode=config.ckpt_mode)
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=config.train_epochs,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
)

trainer.fit(model=model, datamodule=config.datamodule)
