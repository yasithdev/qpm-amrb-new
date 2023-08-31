from typing import Tuple
import einops

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseModel


class Model(BaseModel):
    """
    Classifier with Fisher Exact Randomization Targets

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        in_dims: int,
        rand_perms: int,
        optim_lr: float,
        classifier_loss: str = "crossent",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=False,
        )
        self.in_dims = in_dims
        self.rand_perms = rand_perms
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        K = self.cat_k
        E = self.in_dims
        N = self.rand_perms
        # (K x N) permutation matrix for targets (GT_index = 0)
        self.P = torch.stack([torch.arange(K), *[torch.randperm(K) for _ in range(N - 1)]], dim=1)
        # weight matrix and bias for linear classifier
        self.W = torch.nn.Parameter(torch.randn(E, K, N))
        self.B = torch.nn.Parameter(torch.randn(1, 1, N))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = einops.einsum(x, self.W, "B E, E K N -> B K N") + self.B
        return z

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
        B = y.size(0)
        N = self.rand_perms
        K = self.cat_k
        
        preds = self(x)  # (B, K, N)
        assert list(preds.shape) == [B, K, N]

        targets = self.P.to(y.device)[y, ...]  # (B, N)
        assert list(targets.shape) == [B, N]

        losses = F.cross_entropy(preds, targets, reduction="none") # (B, N)
        assert list(losses.shape) == [B, N]
        
        loss_true = losses[..., 0].mean()
        loss_rand = losses[..., 1:].mean()
        losses_mb["loss_y_true"] = loss_true
        losses_mb["loss_y_rand"] = loss_rand

        # classifier metrics
        pY = preds[..., 0]
        uY = 1 - pY.amax(-1)
        metrics_mb["y_prob"] = pY
        metrics_mb["y_pred"] = pY.argmax(-1)
        metrics_mb["y_ucty"] = uY

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
