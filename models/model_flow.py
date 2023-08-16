from collections import namedtuple
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config

from . import flow
from .base import BaseModel
from .common import edl_loss, edl_probs, margin_loss
from .flow.util import decode_mask


class Model(BaseModel):
    """
    Flow Based Encoder + [Classifier] + (Inverse) Decoder

    """

    def __init__(
        self,
        config: Config,
        with_classifier: bool = True,
        classifier_loss: str = "edl",
        decoder_loss: str = "mse",
    ) -> None:
        super().__init__(
            config=config,
            with_classifier=with_classifier,
            with_decoder=True,
        )
        self.classifier_loss = classifier_loss
        self.decoder_loss = decoder_loss
        hparams = {**locals(), **self.config.as_dict()}
        del hparams["self"]
        del hparams["config"]
        self.save_hyperparameters(hparams)
        self.define_model()
        self.define_metrics()

    def define_model(self):
        config = self.config
        assert config.image_chw

        C, H, W = config.image_chw
        CHW = C * H * W
        D = config.manifold_d
        num_bins = 10
        cm, k0, k1 = 0, 4, 2

        # ambient (x) flow configuration
        c1, h1, w1 = C * k0 * k0, H // k0, W // k0
        c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

        # adjust manifold_d to closest possible value
        cm = D // (h2 * w2)
        D = cm * h2 * w2
        self.config.manifold_d = D
        self.cm = cm

        # categorical configuration
        ind_targets = self.config.get_ind_labels()
        num_labels = len(ind_targets)

        # spatial flow
        arg = namedtuple("arg", ["i_channels", "t_channels", "num_bins"])
        x1_mask = torch.zeros(c1).bool()
        x1_mask[::2] = True
        x1I, x1T = decode_mask(x1_mask)
        rqs_coupling_args_x1A = arg(x1I, x1T, num_bins)
        rqs_coupling_args_x1B = arg(x1T, x1I, num_bins)
        x2_mask = torch.zeros(c2).bool()
        x2_mask[::2] = True
        x2I, x2T = decode_mask(x2_mask)
        rqs_coupling_args_x2A = arg(x2I, x2T, num_bins)
        rqs_coupling_args_x2B = arg(x2T, x2I, num_bins)

        # reduced-channel flow
        u_mask = torch.zeros(cm).bool()
        u_mask[::2] = True
        uI, uT = decode_mask(u_mask)
        rqs_coupling_args_uA = arg(uI, uT, num_bins)
        rqs_coupling_args_uB = arg(uT, uI, num_bins)

        # Base Distributions
        self.dist_z = flow.distributions.StandardNormal(k=D, mu=0.0, std=1.0)
        self.dist_v = flow.distributions.StandardNormal(k=CHW - D, mu=0.0, std=0.01)

        # (B,C,H,W)->(B,16*C,H/4,H/4)->flow->(B,16*C,H/4,H/4)->(B,64*C,H/8,W/8)->flow->(B,64*C,H/8,W/8)->(B,C*H*W)
        self.flow_x = flow.Compose(
            [
                #
                flow.nn.Squeeze(factor=k0),
                flow.nn.ActNorm(c1),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1B._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A._asdict()),
                #
                flow.nn.Squeeze(factor=k1),
                flow.nn.ActNorm(c2),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2B._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A._asdict()),
            ]
        )

        # (B,D) <-> (B,D)
        self.flow_u = flow.Compose(
            [
                #
                flow.nn.ActNorm(cm),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_uA._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_uB._asdict()),
                flow.nn.RQSCoupling(**rqs_coupling_args_uA._asdict()),
                #
                flow.nn.ActNorm(cm),
                #
                flow.nn.Flatten((cm, h2, w2)),
            ]
        )

        # (B,D) -> (B,K)
        if self.with_classifier:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(D, D),
                torch.nn.Tanh(),
                torch.nn.Linear(D, num_labels),
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
        return optimizer

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        uv = self.flow_x(x, forward=True)
        u, v = flow.nn.partition(uv, self.cm)
        z = self.flow_u(u, forward=True)
        uv_m = flow.nn.join(u, v - v)
        x_m = self.flow_x(uv_m, forward=False)
        logits = None
        if self.with_classifier:
            logits = self.classifier(u.flatten(1))
        return u, v, z, x_m, logits

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
        u, v, z, x_m, logits = self(x)

        # classifier loss
        if self.with_classifier:
            if self.classifier_loss == "edl":
                pY, uY = edl_probs(logits)
                losses_mb["loss_y"] = edl_loss(logits, y, self.trainer.current_epoch).mean()
            elif self.classifier_loss == "crossent":
                pY = logits.softmax(-1)
                uY = 1.0 - pY.max(-1)
                losses_mb["loss_y"] = F.cross_entropy(logits, y)
            elif self.classifier_loss == "margin":
                pY = logits.sigmoid()
                uY = 1.0 - pY.max(-1)
                losses_mb["loss_y"] = margin_loss(pY, y).mean()
            else:
                raise ValueError(self.classifier_loss)

            # classifier metrics
            metrics_mb["y_prob"] = pY
            metrics_mb["y_pred"] = pY.argmax(-1)
            metrics_mb["y_ucty"] = uY

        # likelihood losses
        ß = 1e-3
        z_nll = -self.dist_z.log_prob(z.flatten(1))
        v_nll = -self.dist_v.log_prob(v.flatten(1))
        losses_mb["loss_z"] = z_nll.mean()
        losses_mb["loss_v"] = ß * v_nll.mean()

        # manifold losses
        losses_mb["loss_x"] = F.mse_loss(x, x_m)

        # overall metrics
        losses_mb["loss"] = torch.as_tensor(sum(losses_mb.values()))
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
