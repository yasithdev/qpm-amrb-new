import random

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
        source_labels: list[str],
        grouping: list[int],
        labels: list[str],
        in_dims: int,
        rand_perms: int,
        optim_lr: float,
        classifier_loss: str = "crossent",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=len(labels),
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=False,
        )
        self.src_k = len(source_labels)
        self.grouping = grouping
        self.in_dims = in_dims
        self.rand_perms = rand_perms
        self.classifier_loss = classifier_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        S = self.src_k
        K = self.cat_k
        E = self.in_dims
        N = self.rand_perms
        # (N x S) list to build permutations
        P = []
        idxs = list(range(S))
        random.seed(42)
        for _ in range(N):
            P.append(list(map(self.grouping.__getitem__, idxs)))
            random.shuffle(idxs)
        # (S x N) permutation matrix for targets (GT_index = 0)
        self.P = torch.tensor(P).T
        assert list(self.P.shape) == [S, N]
        # weight matrix and bias for linear classifier
        self.W = torch.nn.Parameter(torch.randn(E, K, N))
        self.B = torch.nn.Parameter(torch.randn(1, K, N))

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
        batch: tuple[torch.Tensor, torch.Tensor],
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
        K = self.cat_k
        N = self.rand_perms

        preds = self(x)  # (B, K, N)
        assert list(preds.shape) == [B, K, N]

        targets = self.P.to(y.device)[y, ...]  # (B, N)
        assert list(targets.shape) == [B, N]

        losses = F.cross_entropy(preds, targets, reduction="none")  # (B, N)
        assert list(losses.shape) == [B, N]

        loss_true = losses[..., 0].mean()
        loss_rand = losses[..., 1:].mean()
        losses_mb["loss_y_true"] = loss_true
        losses_mb["loss_y_rand"] = loss_rand

        # classifier metrics
        pY = preds[..., 0].softmax(-1)
        assert list(pY.shape) == [B, K]
        uY = 1 - pY.amax(-1)
        assert list(uY.shape) == [B]
        metrics_mb["y_prob"] = pY  # (B, K)
        metrics_mb["y_pred"] = pY.argmax(-1)  # (B, )
        metrics_mb["y_ucty"] = uY  # (B, )

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
