from functools import partial
from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.optim as optim

from config import Config

from .capsnet.caps import ConvCaps2D, FlattenCaps
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .capsnet.efficientcaps import LinearCapsAR, squash
from .common import Functional, edl_loss, edl_probs, gather_samples, get_conv_out_shape
from .resnet import get_decoder


class Model(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert self.config.image_chw

        ind_targets = self.config.get_ind_labels()
        num_labels = len(ind_targets)
        manifold_d = self.config.manifold_d

        kernel_conv = (9, 9)
        kernel_caps = (3, 3)
        stride_conv = 1
        stride_caps = 2

        # compute output shapes at each layer
        (c, h, w) = self.config.image_chw
        (c1, d1, h1, w1) = 8, 16, h // stride_conv, w // stride_conv
        (c2, d2, h2, w2) = 8, 32, h1 // stride_caps, w1 // stride_caps
        (c3, d3, h3, w3) = 8, 32, h2 // stride_caps, w2 // stride_caps
        (c4, d4, h4, w4) = 8, 32, h3 // stride_caps, w3 // stride_caps

        # (B,C,H,W)->(B,D,K)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=c,
                out_channels=c1 * d1,
                kernel_size=kernel_conv,
                stride=stride_conv,
            ),
            Functional(partial(conv_to_caps, out_capsules=(c1, d1))),
            Functional(squash),
            ConvCaps2D(
                in_capsules=(c1, d1),
                out_capsules=(c1, d1),
                kernel_size=kernel_caps,
                stride=stride_caps,
            ),
            Functional(squash),
            ConvCaps2D(
                in_capsules=(c1, d1),
                out_capsules=(c1, d1),
                kernel_size=kernel_caps,
                stride=stride_caps,
            ),
            Functional(squash),
            ConvCaps2D(
                in_capsules=(c1, d1),
                out_capsules=(c2, d2),
                kernel_size=kernel_caps,
                stride=stride_caps,
            ),
            Functional(squash),
            FlattenCaps(),
            LinearCapsAR(
                in_capsules=(c2, d2 * h4 * w4),
                out_capsules=(manifold_d, num_labels),
            ),
            Functional(squash),
        )

        # (B,D,K)->[(B,K),(B,D)]
        self.classifier = MaskCaps()

        self.decoder = get_decoder(
            num_features=manifold_d,
            output_chw=(c, h, w),
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
        y_z, z_x = self.classifier(z_x)
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
