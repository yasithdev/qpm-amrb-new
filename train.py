import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from config import load_config

# initialize the RNG deterministically
np.random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')

# initialize config, data attributes, and loaders
config = load_config()
config.load_data()
config.print_labels()
assert config.datamodule

wandb_logger = WandbLogger(project="uq_project", log_model="all")
model = config.get_model()
wandb_logger.watch(model, log="all")

checkpoint_callback = ModelCheckpoint(monitor=config.checkpoint_metric, mode="max")
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=config.train_epochs,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
)

trainer.fit(model=model, datamodule=config.datamodule)