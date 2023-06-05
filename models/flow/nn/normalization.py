from typing import Optional, Tuple

import torch
import einops

from . import FlowTransform


class ActNorm(FlowTransform):

    """
    Transform that performs activation normalization. Works for 2D and 4D inputs.
    For 4D inputs (images) normalization is performed per-channel,
    assuming BxCxHxW input shape.

    Reference:
    > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions,
    NeurIPS 2018.

    """

    def __init__(
        self,
        num_features: int,
    ):

        super().__init__()

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = torch.nn.Parameter(torch.zeros(num_features))
        self.shift = torch.nn.Parameter(torch.zeros(num_features))

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4]
        shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)

        # if self.training and not self.initialized:
        #     self._initialize(x)

        scale, shift = self.log_scale.exp().view(shape), self.shift.view(shape)
        z = scale * x + shift

        if x.dim() == 4:
            B, _, H, W = x.size()
            logabsdet = x.new_zeros(B) - self.log_scale.sum() * H * W
        else:
            B, _ = x.size()
            logabsdet = x.new_zeros(B) - self.log_scale.sum()

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4]
        shape = (1, -1, 1, 1) if z.dim() == 4 else (1, -1)

        scale, shift = self.log_scale.exp().view(shape), self.shift.view(shape)
        x = (z - shift) / scale

        if z.dim() == 4:
            B, _, H, W = z.size()
            logabsdet = z.new_zeros(B) + self.log_scale.sum() * H * W
        else:
            B, _ = z.size()
            logabsdet = z.new_zeros(B) + self.log_scale.sum()

        return x, logabsdet

    def _initialize(
        self,
        inputs: torch.Tensor,
    ) -> None:

        """
        Data-dependent initialization, s.t. post-actnorm activations have
        zero mean and unitvariance.

        """
        if inputs.dim() == 4:
            dims = [0, 2, 3]
        else:
            dims = 0

        with torch.no_grad():
            std = inputs.std(dim=dims)
            mu = (inputs / std).mean(dim=dims)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)
