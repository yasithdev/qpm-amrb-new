import random

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseModel


class Model(BaseModel):
    """
    Linear Classifier with Label Permutations for Hypothesis Testing

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        in_dims: int,
        permutations: list[list[int]],
        optim_lr: float,
        classifier_loss: str = "crossent",
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=True,
            with_decoder=False,
            with_permutations=True,
        )
        self.in_dims = in_dims
        self.permutations = permutations
        self.classifier_loss = classifier_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        K = self.cat_k
        E = self.in_dims
        # (S x N) permutation matrix for targets (GT at N=0)
        self.perm = torch.nn.Parameter(torch.tensor(self.permutations), requires_grad=False)
        S, N = self.perm.shape
        # weight matrix and bias for linear classifier
        self.cls_weights = torch.nn.Parameter(torch.randn(E, K, N))
        self.cls_bias = torch.nn.Parameter(torch.randn(1, K, N))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = einops.einsum(x, self.cls_weights, "B E, E K N -> B K N") + self.cls_bias
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
        metrics_mb: dict[str, torch.Tensor] = {"x_true": x}

        # forward pass
        B = y.size(0)
        K = self.cat_k
        S, N = self.perm.shape

        preds = self(x)  # (B, K, N)
        assert list(preds.shape) == [B, K, N]

        # expand y with permutations
        y = self.perm.to(y.device)[y, ...].detach()  # (B, N)
        assert list(y.shape) == [B, N]
        metrics_mb["y_true"] = y[..., 0]
        metrics_mb["y_true_rand"] = y[..., 1:]

        losses = F.cross_entropy(preds, y, reduction="none")  # (B, N)
        assert list(losses.shape) == [B, N]
        losses_mb["loss_y"] = losses[..., 0].mean()
        losses_mb["loss_y_rand"] = losses[..., 1:].mean()

        # classifier metrics
        y_probs = preds.softmax(dim=1)
        pY = y_probs[..., 0]
        pY_rand: torch.Tensor = y_probs[..., 1:]
        assert list(pY.shape) == [B, K]
        uY = 1 - pY.amax(1)
        uY_rand = 1 - pY_rand.amax(1)
        assert list(uY.shape) == [B]
        # metrics - true label
        metrics_mb["y_prob"] = pY  # (B, K)
        metrics_mb["y_pred"] = pY.argmax(1)  # (B, )
        metrics_mb["y_ucty"] = uY  # (B, )
        # metrics - random labels
        metrics_mb["y_prob_rand"] = pY_rand  # (B, K, N-1)
        metrics_mb["y_pred_rand"] = pY_rand.argmax(1)  # (B, N-1)
        metrics_mb["y_ucty_rand"] = uY_rand  # (B, N-1)

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
