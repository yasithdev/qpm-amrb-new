from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.optim as optim

from config import Config

from .common import edl_loss, edl_probs
from .resnet import get_decoder, get_encoder


class Model(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert self.config.image_chw

        ind_targets = self.config.get_ind_labels()
        num_labels = len(ind_targets)

        # (B, C, H, W) -> (B, D, 1, 1)
        self.encoder = get_encoder(
            input_chw=self.config.image_chw,
            num_features=self.config.manifold_d,
        )

        # (B, D, 1, 1) -> (B, C, H, W)
        self.decoder = get_decoder(
            num_features=self.config.manifold_d,
            output_chw=self.config.image_chw,
        )

        # (B, D, 1, 1) -> (B, K)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.config.manifold_d, num_labels),
        )

        # cache variables
        self.y_true = []
        self.y_prob = []
        self.y_ucty = []

        # visualization variables
        self.sample_x_true = None
        self.sample_y_true = None
        self.sample_x_pred = None
        self.sample_y_pred = None
        self.sample_y_ucty = None

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
        return optimizer

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # get batch data
        x, y = batch
        x = x.float()
        y = y.long()

        # forward pass
        z_x = self.encoder(x)
        y_z = self.classifier(z_x)
        x_z = self.decoder(z_x)

        # compute loss
        # L_y_z = margin_loss(y_z, y) - replaced with evidential loss
        pY, uY = edl_probs(y_z.detach())
        L_y_z = edl_loss(y_z, y, self.trainer.current_epoch)
        L_x_z = (x_z - x).pow(2).flatten(1).mean(-1)
        l = 1.0
        L_mb = (L_y_z + l * L_x_z).mean()
        L_y_mb = L_y_z.mean()
        L_x_mb = L_x_z.mean()

        # logging
        self.log("loss_agg", L_mb)
        self.log("loss_y", L_y_mb)
        self.log("loss_x", L_x_mb)

        self.y_true.append(y.detach())
        self.y_prob.append(pY.detach())
        self.y_ucty.append(uY.detach())

        if batch_idx == 0:
            self.sample_x_true = x.detach()
            self.sample_y_true = y.detach()
            self.sample_x_pred = x_z.detach()
            self.sample_y_pred = y_z.detach().argmax(-1)
            self.sample_y_ucty = uY.detach()

        return L_mb

    def on_epoch_end(self):
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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        return self.on_epoch_end()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        return self.on_epoch_end()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        return self.on_epoch_end()
