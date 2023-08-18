from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config

from .base import BaseModel
from .capsnet.caps import FlattenCaps, LinearCapsDR
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .common import Functional, edl_loss, edl_probs, margin_loss
from .resnet import ResidualBlock, get_decoder


class Model(BaseModel):
    """
    CapsNet Encoder + Classifier + [Decoder]

    """

    def __init__(
        self,
        config: Config,
        with_decoder: bool = True,
        classifier_loss: str = "margin",
        decoder_loss: str = "mse",
    ) -> None:
        super().__init__(
            config=config,
            with_classifier=True,
            with_decoder=with_decoder,
        )
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        hparams = {**locals(), **self.config.as_dict()}
        del hparams["self"]
        del hparams["config"]
        self.save_hyperparameters(hparams)
        self.define_model()
        self.define_metrics()

    def define_model(self):
        assert self.config.image_chw
        # params
        K = len(self.ind_labels)
        (C, H, W) = self.config.image_chw
        D = self.config.manifold_d
        # (B, C, H, W) -> (B, D, K)
        self.encoder = torch.nn.Sequential(
            ResidualBlock(
                in_channels=C,
                hidden_channels=C * 4,
                out_channels=C * 4,
                stride=2,
                conv=torch.nn.Conv2d,
            ),
            ResidualBlock(
                in_channels=C * 4,
                hidden_channels=C * 8,
                out_channels=C * 8,
                stride=2,
                conv=torch.nn.Conv2d,
            ),
            ResidualBlock(
                in_channels=C * 8,
                hidden_channels=C * 16,
                out_channels=C * 16,
                stride=2,
                conv=torch.nn.Conv2d,
            ),
            Functional(partial(conv_to_caps, out_capsules=(16, C))),
            FlattenCaps(),
            LinearCapsDR(
                in_capsules=(16, C * H * W // 64),
                out_capsules=(D, K),
            ),
        )
        # (B, D, K) -> (B, K)
        self.classifier = MaskCaps()
        # (B, D, 1, 1) -> (B, C, H, W)
        if self.with_decoder:
            self.decoder = get_decoder(
                num_features=D,
                output_chw=self.config.image_chw,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.encoder(x)
        logits, zi = self.classifier(z)
        x_pred = None
        if self.with_decoder:
            x_pred = self.decoder(zi)
        return z, logits, zi, x_pred

    def compute_losses(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        # get batch data
        x, y = batch
        x = x.float()
        y = y.long()

        # accumulators
        losses_mb: dict[str, torch.Tensor] = {}
        metrics_mb: dict[str, torch.Tensor] = {"x_true": x, "y_true": y}

        # forward pass
        z, logits, zi, x_pred = self(x)

        # classifier loss
        if self.classifier_loss == "edl":
            pY, uY = edl_probs(logits)
            losses_mb["loss_y"] = edl_loss(logits, y, self.trainer.current_epoch).mean()
        elif self.classifier_loss == "crossent":
            pY = logits.softmax(-1)
            uY = 1 - pY.amax(-1)
            losses_mb["loss_y"] = F.cross_entropy(logits, y)
        elif self.classifier_loss == "margin":
            pY = logits.sigmoid()
            uY = 1 - pY.amax(-1)
            losses_mb["loss_y"] = margin_loss(pY, y).mean()
        else:
            raise ValueError(self.classifier_loss)

        # classifier metrics
        metrics_mb["y_prob"] = pY
        metrics_mb["y_pred"] = pY.argmax(-1)
        metrics_mb["y_ucty"] = uY

        # decoder loss
        if self.with_decoder:
            assert x_pred is not None
            if self.decoder_loss == "mse":
                L_x = F.mse_loss(x_pred, x)
            else:
                raise ValueError(self.decoder_loss)
            l = 1.0
            losses_mb["loss_x"] = l * L_x.mean()

            # decoder metrics
            metrics_mb["x_pred"] = x_pred

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
