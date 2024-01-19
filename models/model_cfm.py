from typing import Literal, Tuple
import einops

import torch
import torch.optim as optim
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE

from .base import BaseModel


class Model(BaseModel):
    """
    Conditional Flow Model

    """

    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        emb_dims: int,
        ignore_dims: Literal["channel", "spatial", "checkerboard"] | None,
        input_shape: tuple[int, int, int],
        optim_lr: float,
        strategy: str = "otcfm",
        unit_sphere: bool = False,
    ) -> None:
        super().__init__(
            labels=labels,
            cat_k=cat_k,
            optim_lr=optim_lr,
            with_classifier=False,
            with_decoder=True,
        )
        self.input_shape = input_shape
        self.ignore_dims = ignore_dims
        self.unit_sphere = unit_sphere
        c, h, w = input_shape
        self.mask = torch.nn.Parameter(torch.zeros(self.input_shape), requires_grad=False)

        if self.ignore_dims == "spatial":
            assert emb_dims % c == 0, f"emb_dims ({emb_dims}) not a multiple of c ({c})"
            spatial_dims = emb_dims // c
            manifold_h = int((spatial_dims * h / w) ** 0.5)
            manifold_w = int((spatial_dims * w / h) ** 0.5)
            assert manifold_h * manifold_w == spatial_dims
            # compute indices
            lo_c, lo_h, lo_w = 0, (h - manifold_h) // 2, (w - manifold_w) // 2
            hi_c, hi_h, hi_w = c, lo_h + manifold_h, lo_w + manifold_w
            # update mask
            self.mask[lo_c : hi_c, lo_h : hi_h, lo_w : hi_w] = 1
        if self.ignore_dims == "channel":
            assert emb_dims % (h * w) == 0, "emb_dims not a multiple of spatial size"
            manifold_c = emb_dims // (h * w)
            # compute indices
            lo_c, lo_h, lo_w = 0, 0, 0
            hi_c, hi_h, hi_w = manifold_c, h, w
            # update mask
            self.mask[lo_c : hi_c, lo_h : hi_h, lo_w : hi_w] = 1
        if self.ignore_dims == "checkerboard":
            assert emb_dims % c == 0, f"emb_dims ({emb_dims}) not a multiple of c ({c})"
            spatial_dims = emb_dims // c
            manifold_h = int((spatial_dims * h / w) ** 0.5)
            manifold_w = int((spatial_dims * w / h) ** 0.5)
            assert manifold_h * manifold_w == spatial_dims
            # compute indices
            h_idxs = torch.linspace(0, h-1, manifold_h).int().tolist()
            w_idxs = torch.linspace(0, w-1, manifold_w).int().tolist()
            # update mask
            for ih in h_idxs:
                for iw in w_idxs:
                    self.mask[:,ih, iw] = 1
        if self.ignore_dims is None:
            # compute indices
            lo_c, lo_h, lo_w = 0, 0, 0
            hi_c, hi_h, hi_w = c, h, w
            # update mask
            self.mask[lo_c : hi_c, lo_h : hi_h, lo_w : hi_w] = 1

        self.strategy = strategy
        self.save_hyperparameters()
        self.define_model()
        self.define_metrics()

    def define_model(self):
        # params
        sigma = 0.0

        self.net_model = UNetModelWrapper(
            dim=self.input_shape,
            num_res_blocks=3,
            num_channels=64,
            channel_mult=[1, 1, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        )

        if self.strategy == "otcfm":
            self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif self.strategy == "cfm":
            self.fm = ConditionalFlowMatcher(sigma=sigma)
        elif self.strategy == "fm":
            self.fm = TargetConditionalFlowMatcher(sigma=sigma)
        else:
            raise NotImplementedError(self.strategy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optim_lr)
        lrs = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
        return [optimizer], [lrs]

    def forward(
        self,
        x: torch.Tensor,
        τ: float = 0.0,
    ) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            x_base = torch.rand_like(x) * self.mask
        # alternative - sample from unit sphere instead
        if self.unit_sphere:
            x /= x.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        t, xt, ut, *_ = self.fm.sample_location_and_conditional_flow(x_base, x)
        vt = self.net_model(t, xt)
        return t, xt, ut, vt, x_base

    def ema(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
        decay: float,
    ):
        source_dict: dict[str, torch.nn.Parameter] = source.state_dict()
        target_dict: dict[str, torch.nn.Parameter] = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))

    def generate_samples(
        self,
        x_base: torch.Tensor,
        rescale: bool = False,
    ) -> torch.Tensor:
        self.net_model.eval()
        with torch.no_grad():
            ode = NeuralODE(self.net_model, solver="dopri5", sensitivity="adjoint")
            t_span = torch.linspace(0, 1, 100)
            traj = ode.trajectory(x_base, t_span)
        traj = traj[-1, :].view(x_base.shape)
        if rescale:
            traj = traj.clip(-1, 1) / 2 + 0.5
        self.net_model.train()
        return traj

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
        metrics_mb: dict[str, torch.Tensor] = {}

        # forward pass
        τ = 0.0
        if stage == "train":
            # 1.0 upto epoch 100, anneal to 0.0 in 100 epochs
            τ = 1.0 - min(max(self.current_epoch - 100, 0) / 100, 1.0)
        _, xt, ut, vt, x_base = self(x, τ)
        loss = torch.mean((vt - ut) ** 2)

        # manifold metrics (only first batch)
        if batch_idx == 0:
            metrics_mb["x_true"] = x
            metrics_mb["y_true"] = y
            metrics_mb["x_pred"] = self.generate_samples(x_base)

        # overall metrics
        losses_mb["loss"] = loss
        self.log_losses(stage, losses_mb)
        self.update_metrics(stage, batch_idx, metrics_mb)

        # return minibatch loss
        return losses_mb["loss"]
