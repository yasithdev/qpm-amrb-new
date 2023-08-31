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
        # (N x K) permutation matrix for targets (GT_index = 0)
        self.P = torch.stack([torch.arange(K), *[torch.randperm(K) for _ in range(N - 1)]], dim=0)
        # weight matrix and bias for linear classifier
        self.W = torch.nn.Parameter(torch.randn(N, E, K))
        self.B = torch.nn.Parameter(torch.randn(N, 1, 1))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = einops.einsum(x, self.W, "B E, N E K -> N B K") + self.B
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
        candidates = self(x)  # (N, B, K)
        targets = self.P.to(y.device)[:, y]  # (N, B)
        assert list(candidates.shape) == [N, B, K]
        assert list(targets.shape) == [N, B]

        # classifier loss
        losses = F.cross_entropy(candidates.reshape(-1, K), targets.reshape(-1), reduction="none").reshape(N, B).mean(-1)
        assert list(losses.shape) == [N]
        
        loss_true = losses[0]
        loss_rand = losses[1:].mean()
        losses_mb["loss_y_true"] = loss_true
        losses_mb["loss_y_rand"] = loss_rand

        # classifier metrics
        pY = candidates[0]
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
