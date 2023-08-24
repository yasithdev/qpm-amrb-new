from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from .common import simclr_loss, edl_loss, edl_probs, margin_loss

from .base import BaseModel


class Model(BaseModel):
    """
    ResNet50 Encoder + Classifier (No Decoder)

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        manifold_d: int,
        image_chw: tuple[int, int, int],
        optim_lr: float,
        temperature: float,
        with_classifier: bool = False,
        encoder_loss: str = "simclr",
        classifier_loss: str = "edl",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            manifold_d=manifold_d,
            image_chw=image_chw,
            optim_lr=optim_lr,
            with_classifier=with_classifier,
            with_decoder=False,
        )
        self.temperature = temperature
        self.encoder_loss = encoder_loss
        self.classifier_loss = classifier_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        D = self.manifold_d
        K = self.cat_k
        # pretrained resnet50 model
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        # (B, C, H, W) -> (B, D)
        self.f = torch.nn.Sequential(
            *list(model.children())[:-1],
        )
        # (B, D) -> (B, K)
        self.g1 = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
        )
        self.g2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True)
        )
        self.g3 = torch.nn.Linear(512, D, bias=True)

        self.classifier = None
        if self.with_classifier:
            # two-layer classifier with ReLU in between
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(D, D),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(D, K),
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        lrs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lrs
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    
        feature = self.f(x).flatten(start_dim=1)
        proj_1 = self.g1(feature)
        proj_2 = self.g2(proj_1)
        proj_3 = self.g3(proj_2)

        logits = None
        if self.with_classifier:
            assert self.classifier is not None
            logits = self.classifier(proj_1)
        
        # return unit vectors (TODO clarify)
        return F.normalize(proj_1, dim=-1), F.normalize(proj_3, dim=-1), logits

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

        if self.encoder_loss == "simclr":
            # assuming x is a tuple
            # can also call transform(x) here twice and get 2 crops
            (x_a, x_b) = x
            assert list(x_a.shape) == list(x_b.shape)
            _, proj_a, logits = self(x_a)
            _, proj_b, logits = self(x_b)
            losses_mb["loss_simclr"] = simclr_loss(proj_a, proj_b, self.temperature)
        else:
            raise ValueError()
        
        if self.with_classifier:
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
