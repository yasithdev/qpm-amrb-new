from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from .base import BaseModel
from .common import edl_loss, edl_probs, margin_loss


class Model(BaseModel):
    """
    ResNet18 Encoder + Classifier (No Decoder)

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        emb_dims: int,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        classifier_loss: str = "edl",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=False,
        )
        self.emb_dims = emb_dims
        self.input_shape = input_shape
        self.classifier_loss = classifier_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        K = self.cat_k
        # pretrained resnet18 model
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        E = model.fc.in_features
        del model.fc
        # (B, C, H, W) -> (B, D)
        self.encoder = model
        # (B, D) -> (B, K)
        self.classifier = torch.nn.Linear(E, K)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)
        return z, logits

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
        z, logits = self(x)

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

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
