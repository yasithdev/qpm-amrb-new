from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseModel
from .common import edl_loss, edl_probs, margin_loss
from .resnet import get_decoder, get_encoder


class Model(BaseModel):
    """
    ResNet Encoder + Classifier + [Decoder]

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        emb_dims: int,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        with_decoder: bool = True,
        classifier_loss: str = "edl",
        decoder_loss: str = "mse",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=with_decoder,
        )
        self.emb_dims = emb_dims
        self.input_shape = input_shape
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        K = self.cat_k
        # (B, C, H, W) -> (B, D, 1, 1)
        self.encoder = get_encoder(
            input_chw=self.input_shape,
            num_features=self.emb_dims,
        )
        # (B, D, 1, 1) -> (B, K)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.emb_dims, K),
        )
        # (B, D, 1, 1) -> (B, C, H, W)
        if self.with_decoder:
            self.decoder = get_decoder(
                num_features=self.emb_dims,
                output_chw=self.input_shape,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.encoder(x)
        logits = self.classifier(z)
        x_pred = None
        if self.with_decoder:
            x_pred = self.decoder(z)
        return z, logits, x_pred

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
        z, logits, x_pred = self(x)

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
