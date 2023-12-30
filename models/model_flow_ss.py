from collections import namedtuple
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from . import flow
from .base import BaseModel
from .common import edl_loss, edl_probs, margin_loss, vcreg_loss
from .flow.util import decode_mask


class Model(BaseModel):
    """
    Flow Based Encoder + [Classifier] + (Inverse) Decoder

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        emb_dims: int,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        with_classifier: bool = False,
        classifier_loss: str = "edl",
        x_loss: str = "mse",
        u_loss: str = "vcr",
        v_loss: str = "mse",
    ) -> None:
        """
        Initialize Single-Scale Model

        Args:
            labels (list[str]): list of labels
            cat_k (int): number of classes
            emb_dims (int): size of feature dims
            input_shape (tuple[int, int, int]): input shape C,H,W
            optim_lr (float): learning rate
            with_classifier (bool, optional): True if including classifier. Defaults to False.
            classifier_loss (str, optional): Allowed - edl, crossent, margin, N/A. Defaults to "edl".
            x_loss (str, optional): Allowed - mse, N/A. Defaults to "mse".
            u_loss (str, optional): Allowed - nll, vcr, N/A. Defaults to "vcr".
            v_loss (str, optional): Allowed - nll, l1, mse, N/A. Defaults to "mse".
        """
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=with_classifier,
            with_decoder=True,
        )
        self.emb_dims = emb_dims
        self.input_shape = input_shape
        self.classifier_loss = classifier_loss
        self.x_loss = x_loss
        self.u_loss = u_loss
        self.v_loss = v_loss
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        # params
        K = self.cat_k
        D = self.emb_dims
        C, H, W = self.input_shape
        num_bins = 5
        if self.input_shape[-1] >= 224:  # (C,224,224) -> (64*C,28,28) -> (1024*C,7,7)
            assert self.input_shape[-1] % 32 == 0
            k0, k1 = 8, 4
        else:  # (C,32,32) -> (16*C,8,8) -> (256*C,2,2)
            assert self.input_shape[-1] % 8 == 0
            k0, k1 = 4, 4

        # ambient (x) flow configuration
        c1, h1, w1 = C * k0 * k0, H // k0, W // k0
        c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

        # spatial flow
        arg = namedtuple("arg", ["in_h", "in_w", "i_channels", "t_channels", "num_bins"])
        x1_mask = torch.zeros(c1).bool()
        x1_mask[::2] = True
        x1I, x1T = decode_mask(x1_mask)
        rqs_coupling_args_x1A = arg(h1, w1, x1I, x1T, num_bins)
        rqs_coupling_args_x1B = arg(h1, w1, x1T, x1I, num_bins)
        x2_mask = torch.zeros(c2).bool()
        x2_mask[::2] = True
        x2I, x2T = decode_mask(x2_mask)
        rqs_coupling_args_x2A = arg(h2, w2, x2I, x2T, num_bins)
        rqs_coupling_args_x2B = arg(h2, w2, x2T, x2I, num_bins)

        # (B, C, H, W) -> ... -> (B, CHW, 1, 1)
        assert h2 == w2
        self.flow_x = flow.Compose(
            [
                #
                flow.nn.Squeeze(factor=k0),
                flow.nn.ActNorm(c1),
                flow.nn.ConformalConv2D(c1),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A._asdict(), spatial=True),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1B._asdict(), spatial=True),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A._asdict(), spatial=True),
                #
                flow.nn.Squeeze(factor=k1),
                flow.nn.ActNorm(c2),
                flow.nn.ConformalConv2D(c2),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A._asdict(), spatial=True),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2B._asdict(), spatial=True),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A._asdict(), spatial=True),
                #
                flow.nn.Squeeze(factor=h2),  # get to (1, 1) spatial size
                #
            ]
        )

        # (B,D) -> (B,K)
        if self.with_classifier:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(D, D),
                torch.nn.Tanh(),
                torch.nn.Linear(D, K),
            )

        # Base Distributions
        self.dist_u = flow.distributions.StandardNormal(k=D, mu=0.0, std=1.0)
        self.dist_v = flow.distributions.StandardNormal(k=C * H * W - D, mu=0.0, std=0.01)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor | None]:
        uv, logabsdet_x = self.flow_x(x, forward=True)
        u, v = flow.nn.partition(uv, self.emb_dims)
        uv_m = flow.nn.join(u, torch.zeros_like(v))  # concatenate u with zero vector
        x_m, logabsdet_m = self.flow_x(uv_m, forward=False)
        u, v = u.flatten(1), v.flatten(1)
        u_norm = u.norm(dim=1, keepdim=True)
        v_norm = v.norm(dim=1, keepdim=True)
        logits = None
        if self.with_classifier:
            logits = self.classifier(u)
        return (u, v), (u_norm, v_norm), x_m, logits

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
        (u, v), (u_norm, v_norm), x_m, logits = self(x)

        # classifier loss
        if self.with_classifier:
            assert logits is not None
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

        # manifold losses
        w_x, w_u, w_v = 1.0, 0.1, 0.1
        ## x
        if self.x_loss == "mse":
            losses_mb["loss_x"] = w_x * F.mse_loss(x_m, x)
        elif self.x_loss == "N/A":
            pass
        else:
            raise ValueError(self.x_loss)
        ## u
        if self.u_loss == "vcr":
            losses_mb["loss_u"] = w_u * vcreg_loss(u)
        if self.u_loss == "nll":
            losses_mb["loss_u"] = w_u * -self.dist_u.log_prob(u).mean()
        elif self.u_loss == "N/A":
            pass
        else:
            raise ValueError(self.u_loss)
        ## v
        if self.v_loss == "nll":
            losses_mb["loss_v"] = w_v * -self.dist_v.log_prob(v).mean()
        elif self.v_loss == "l1":
            losses_mb["loss_v"] = w_v * F.l1_loss(v, torch.zeros_like(v))
        elif self.v_loss == "mse":
            losses_mb["loss_v"] = w_v * F.mse_loss(v, torch.zeros_like(v))
        elif self.v_loss == "N/A":
            pass
        else:
            raise ValueError(self.v_loss)

        # manifold metrics
        metrics_mb["x_pred"] = x_m

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
