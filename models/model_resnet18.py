from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from config import Config

from .base import BaseModel
from .common import edl_loss, edl_probs, margin_loss


class Model(BaseModel):
    """
    ResNet18 Encoder + Classifier (No Decoder)

    """

    def __init__(
        self,
        config: Config,
        classifier_loss: str = "edl",
    ) -> None:
        super().__init__(
            config=config,
            with_classifier=True,
            with_decoder=False,
        )
        self.config = config
        self.classifier_loss = classifier_loss
        hparams = {**locals(), **self.config.as_dict()}
        del hparams["self"]
        del hparams["config"]
        self.save_hyperparameters(hparams)
        self.define_model()
        self.define_metrics()

    def define_model(self):
        assert self.config.image_chw
        K = len(self.ind_labels)
        # pretrained resnet18 model
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        # (B, C, H, W) -> (B, D)
        self.encoder = torch.nn.Sequential(
            *list(model.children())[:-1],
        )
        # (B, D) -> (B, K)
        self.classifier = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=K,
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
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
        x, y = batch
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
