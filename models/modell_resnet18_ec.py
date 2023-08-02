from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.optim as optim
import torchvision.models as models

from config import Config

from .common import edl_loss, edl_probs


class Model(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert self.config.image_chw

        ind_targets = self.config.get_ind_labels()
        num_labels = len(ind_targets)

        # (B, C, H, W) -> (B, D, 1, 1)
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features, num_labels)

        # cache variables
        self.y_true = []
        self.y_prob = []
        self.y_ucty = []

        # visualization variables
        self.sample_x_true = None
        self.sample_y_true = None
        self.sample_y_pred = None
        self.sample_y_ucty = None

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # get batch data
        x, y = batch
        x = x.float()
        y = y.long()

        # forward pass
        y_x = self.encoder(x)

        # compute loss
        # L_y_z = margin_loss(y_z, y) - replaced with evidential loss
        pY, uY = edl_probs(y_x.detach())
        L_y_z = edl_loss(y_x, y, self.trainer.current_epoch)
        L_mb = L_y_z.mean()
        L_y_mb = L_y_z.mean()

        # logging
        self.log("loss_agg", L_mb)
        self.log("loss_y", L_y_mb)

        self.y_true.append(y.detach().cpu())
        self.y_prob.append(pY.detach().cpu())
        self.y_ucty.append(uY.detach().cpu())

        if batch_idx == 0:
            self.sample_x_true = x.detach().cpu()
            self.sample_y_true = y.detach().cpu()
            self.sample_y_pred = y_x.detach().argmax(-1).cpu()
            self.sample_y_ucty = uY.detach().cpu()

        return L_mb

    def on_train_epoch_end(self):
        # concatenate
        y_true = torch.cat(self.y_true)
        y_prob = torch.cat(self.y_prob)
        y_ucty = torch.cat(self.y_ucty)
        
        # log tensors
        ...

        # log samples
        ...

        # cleanup
        self.y_true.clear()
        self.y_prob.clear()
        self.y_ucty.clear()
