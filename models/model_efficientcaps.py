from collections import namedtuple
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseModel
from .capsnet.caps import ConvCaps2D, FlattenCaps
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .capsnet.efficientcaps import LinearCapsAR, squash
from .common import Functional, edl_loss, edl_probs, margin_loss
from .resnet import get_decoder


class Model(BaseModel):
    """
    CapsNet Encoder + Classifier + [Decoder]

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        manifold_d: int,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        with_decoder: bool = True,
        classifier_loss: str = "edl",
        decoder_loss: str = "margin",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=with_decoder,
        )
        self.manifold_d = manifold_d
        self.input_shape = input_shape
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        # params
        K = self.cat_k
        D = self.manifold_d
        kernel_conv = (7, 7)
        kernel_caps = (3, 3)
        stride_conv = 1
        stride_caps = 2

        # compute output shapes at each layer
        (c, h, w) = self.input_shape
        (c1, d1, h1, w1) = 8, 16, h // stride_conv, w // stride_conv
        (c2, d2, h2, w2) = 8, 32, h1 // stride_caps, w1 // stride_caps
        (c3, d3, h3, w3) = 8, 32, h2 // stride_caps, w2 // stride_caps
        (c4, d4, h4, w4) = 8, 32, h3 // stride_caps, w3 // stride_caps

        # (B, C, H, W) -> (B, D, K)
        arg_conv = namedtuple("arg_conv", ["in_channels", "out_channels", "kernel_size", "stride"])
        arg_caps = namedtuple("arg_caps", ["in_capsules", "out_capsules", "kernel_size", "stride"])
        arg_rout = namedtuple("arg_rout", ["in_capsules", "out_capsules"])
        self.encoder = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Conv2d(**arg_conv(c, c1 * d1, kernel_conv, stride_conv)._asdict()),
                Functional(partial(conv_to_caps, out_capsules=(c1, d1))),
                Functional(squash),
            ),
            torch.nn.Sequential(
                ConvCaps2D(**arg_caps((c1, d1), (c1, d1), kernel_caps, stride_caps)._asdict()),
                Functional(squash),
            ),
            torch.nn.Sequential(
                ConvCaps2D(**arg_caps((c1, d1), (c1, d1), kernel_caps, stride_caps)._asdict()),
                Functional(squash),
            ),
            torch.nn.Sequential(
                ConvCaps2D(**arg_caps((c1, d1), (c2, d2), kernel_caps, stride_caps)._asdict()),
                Functional(squash),
            ),
            FlattenCaps(),
            LinearCapsAR(**arg_rout((c2, d2 * h4 * w4), (D, K))._asdict()),
        )
        # (B, D, K) -> (B, K), (B, D)
        self.classifier = MaskCaps()

        # (B, D, 1, 1) -> (B, C, H, W)
        if self.with_decoder:
            self.decoder = get_decoder(
                num_features=D,
                output_chw=(c, h, w),
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
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
        x, y, *_ = batch
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
