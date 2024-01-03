from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from config import Config

from .common import Functional, simclr_loss, vicreg_loss, edl_loss, edl_probs, margin_loss
from datasets.transforms import simclr_transform

from .base import BaseModel


class Model(BaseModel):
    """
    ResNet50 Encoder + Classifier (No Decoder)

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        emb_dims: int,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        opt: Config,
        with_classifier: bool = True,
        with_embedder: bool = False,
        embedding_loss: str = "vicreg",
        classifier_loss: str = "edl",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=with_classifier,
            with_decoder=False,
        )
        self.emb_dims = emb_dims
        self.input_shape = input_shape
        self.temperature = opt.temperature
        self.with_embedder = with_embedder
        self.embedding_loss = embedding_loss
        self.classifier_loss = classifier_loss
        self.augment_fn = {
            "train": simclr_transform(opt, eval=False),
            "val": simclr_transform(opt, eval=opt.train_supervised),
            "test": simclr_transform(opt, eval=True),
            "predict": simclr_transform(opt, eval=True),
        }
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        D = self.emb_dims
        K = self.cat_k
        # pretrained resnet50 model
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        E = model.fc.in_features
        model.fc = torch.nn.Identity() # type: ignore
        # (B, C, H, W) -> (B, E)
        self.encoder = model
        # (B, E) -> (B, D)
        self.embedder = None
        if self.with_embedder:
            self.embedder = torch.nn.Sequential(
                torch.nn.Sequential(
                    torch.nn.Linear(E, 1024, bias=False),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(inplace=True),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(1024, 512, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),
                ),
                torch.nn.Linear(512, D, bias=True),
                Functional(lambda z : F.normalize(z, dim=-1)),
            )
        # (B, E) -> (B, K)
        self.classifier = None
        if self.with_classifier:
            # two-layer classifier with ReLU in between
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(E, D),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(D, K),
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        lrs = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return {"optimizer": optimizer, "lr_scheduler": lrs}

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        feature = self.encoder(x).flatten(start_dim=1)
        embedding = None
        if self.with_embedder:
            assert self.embedder is not None
            embedding = self.embedder(feature)
        logits = None
        if self.with_classifier:
            assert self.classifier is not None
            logits = self.classifier(feature)
        return feature, embedding, logits

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

        # embedder loss
        if self.with_embedder:
            T = self.temperature
            # get 2 views of x
            xA = self.augment_fn[stage](x)
            xB = self.augment_fn[stage](x)
            # forward pass on each view
            zA, eA, cA = self(xA)
            zB, eB, cB = self(xB)
            if self.embedding_loss == "simclr":
                emb_loss = simclr_loss([eA, eB], temperature=T)
            elif self.embedding_loss == "vicreg":
                emb_loss = vicreg_loss(eA, eB)
            else:
                raise ValueError(self.embedding_loss)
            # embedder metrics
            losses_mb["loss_emb"] = emb_loss
            metrics_mb["loss_emb"] = emb_loss

        # classifier loss
        if self.with_classifier:
            # forward pass
            z, e, l = self(x)
            if self.classifier_loss == "edl":
                pY, uY = edl_probs(l)
                losses_mb["loss_y"] = edl_loss(l, y, self.trainer.current_epoch).mean()
            elif self.classifier_loss == "crossent":
                pY = l.softmax(-1)
                uY = 1 - pY.amax(-1)
                losses_mb["loss_y"] = F.cross_entropy(l, y)
            elif self.classifier_loss == "margin":
                pY = l.sigmoid()
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
