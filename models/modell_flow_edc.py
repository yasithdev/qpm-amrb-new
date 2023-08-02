from functools import partial
from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.optim as optim

from config import Config

from . import flow
from .common import edl_loss, edl_probs, gather_samples
from .flow.util import decode_mask


class Model(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert self.config.image_chw

        C, H, W = config.image_chw
        CHW = C * H * W
        D = config.manifold_d
        num_bins = 10

        # ambient (x) flow configuration
        c1, h1, w1 = C * k0 * k0, H // k0, W // k0
        c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

        # adjust manifold_d to closest possible value
        cm, k0, k1 = 0, 4, 2
        cm = D // (h2 * w2)
        D = cm * h2 * w2
        self.config.manifold_d = D

        # categorical configuration
        ind_targets = self.config.get_ind_labels()
        num_labels = len(ind_targets)

        # spatial flow
        x1_mask = torch.zeros(c1).bool()
        x1_mask[::2] = True
        x1I, x1T = decode_mask(x1_mask)
        rqs_coupling_args_x1A = {
            "i_channels": x1I,
            "t_channels": x1T,
            "num_bins": num_bins,
        }
        rqs_coupling_args_x1B = {
            "i_channels": x1T,
            "t_channels": x1I,
            "num_bins": num_bins,
        }

        x2_mask = torch.zeros(c2).bool()
        x2_mask[::2] = True
        x2I, x2T = decode_mask(x2_mask)
        rqs_coupling_args_x2A = {
            "i_channels": x2I,
            "t_channels": x2T,
            "num_bins": num_bins,
        }
        rqs_coupling_args_x2B = {
            "i_channels": x2T,
            "t_channels": x2I,
            "num_bins": num_bins,
        }

        # reduced-channel flow
        u_mask = torch.zeros(cm).bool()
        u_mask[::2] = True
        uI, uT = decode_mask(u_mask)
        rqs_coupling_args_uA = {
            "i_channels": uI,
            "t_channels": uT,
            "num_bins": num_bins,
        }
        rqs_coupling_args_uB = {
            "i_channels": uT,
            "t_channels": uI,
            "num_bins": num_bins,
        }

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
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
                flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                #
                flow.nn.Squeeze(factor=k1),
                flow.nn.ActNorm(c2),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
                flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
            ]
        )

        # (B,D) <-> (B,D)
        self.flow_u = flow.Compose(
            [
                #
                flow.nn.ActNorm(cm),
                #
                flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                flow.nn.RQSCoupling(**rqs_coupling_args_uB),
                flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                #
                flow.nn.ActNorm(cm),
                #
                flow.nn.Flatten((cm, h2, w2)),
            ]
        )

        # (B,D) -> (B,K)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(D, D),
            torch.nn.Tanh(),
            torch.nn.Linear(D, num_labels),
        )

        # cache variables
        self.y_true = []
        self.y_prob = []
        self.y_ucty = []
        self.u_norm = []
        self.v_norm = []
        self.z_norm = []
        self.z_nll = []

        # visualization variables
        self.sample_x_true = None
        self.sample_y_true = None
        self.sample_x_pred = None
        self.sample_y_pred = None
        self.sample_y_ucty = None

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.optim_lr)
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # get batch data
        x, y = batch
        x = x.float()
        y = y.long()

        # forward pass
        # x -> (u x v) -> u -> z
        uv_x, _ = flow_x(x, forward=True)
        u_x, v_x = flow.nn.partition(uv_x, self.cm)
        z_x, _ = flow_u(u_x, forward=True)
        # u -> (u x 0) -> x~
        u_z = u_x
        v_z = v_x.new_zeros(v_x.size())
        uv_z = flow.nn.join(u_z, v_z)
        x_z, _ = flow_x(uv_z, forward=False)
        # u -> y
        y_z = classifier(u_x.flatten(1))
        pY, uY = edl_probs(y_z.detach())

        # compute loss
        L_z = -dist_z.log_prob(z_x.flatten(1))
        L_v = -dist_v.log_prob(v_x.flatten(1))
        L_x = (x - x_z).flatten(1).pow(2).sum(-1)
        # L_y_z = F.cross_entropy(y_x, y) - used EDL alternative
        L_y_z = edl_loss(y_z, y, epoch)
        ß = 1e-3
        # L_mb = L_z.mean() + (ß * L_v.mean()) + L_m.mean() + L_y_z.mean() # Σ_i[Σ_n(L_{n,i})/N]
        L_mb = (L_z + (ß * L_v) + L_x + L_y_z).mean()  # Σ_n[Σ_i(L_{n,i})]/N
        L_y_mb = L_y_z.mean()
        L_x_mb = L_x.mean()
        L_v_mb = L_v.mean()
        L_z_mb = L_z.mean()

        # logging
        self.log("loss_agg", L_mb)
        self.log("loss_y", L_y_mb)
        self.log("loss_x", L_x_mb)
        self.log("loss_v", L_v_mb)
        self.log("loss_z", L_z_mb)

        self.y_true.append(y.detach().cpu())
        self.y_prob.append(pY.detach().cpu())
        self.y_ucty.append(uY.detach().cpu())
        self.u_norm.append(u_x.detach().flatten(1).norm(2, -1).cpu())
        self.v_norm.append(v_x.detach().flatten(1).norm(2, -1).cpu())
        self.z_norm.append(z_x.detach().flatten(1).norm(2, -1).cpu())
        self.z_nll.append(L_z.detach().cpu())

        if batch_idx == 0:
            self.sample_x_true = x.detach().cpu()
            self.sample_y_true = y.detach().cpu()
            self.sample_x_pred = x_z.detach().cpu()
            self.sample_y_pred = y_x.detach().argmax(-1).cpu()
            self.sample_y_ucty = uY.detach().cpu()

        return L_mb

    def on_train_epoch_end(self):
        # concatenate
        y_true = torch.cat(self.y_true)
        y_prob = torch.cat(self.y_prob)
        y_ucty = torch.cat(self.y_ucty)
        u_norm = torch.cat(self.u_norm)
        v_norm = torch.cat(self.v_norm)
        z_norm = torch.cat(self.z_norm)
        z_nll = torch.cat(self.z_nll)

        # log tensors
        ...

        # log samples
        ...

        # cleanup
        self.y_true.clear()
        self.y_prob.clear()
        self.y_ucty.clear()
        self.u_norm.clear()
        self.v_norm.clear()
        self.z_norm.clear()
        self.z_nll.clear()
