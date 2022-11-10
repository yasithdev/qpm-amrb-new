from typing import Optional, Tuple

import einops
import torch

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

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        print(x.size())
        assert x.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self.scale.view(view_shape), self.shift.view(view_shape)
        outputs = scale * x + shift

        if x.dim() == 4:
            batch_size, _, h, w = x.shape
            logabsdet = h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = x.shape
            logabsdet = torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if z.dim() == 4 else (1, -1)

        scale, shift = self.scale.view(view_shape), self.shift.view(view_shape)
        outputs = (z - shift) / scale

        if z.dim() == 4:
            batch_size, _, h, w = z.shape
            logabsdet = (
                -h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
            )
        else:
            batch_size, _ = z.shape
            logabsdet = -torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(
        self,
        inputs,
    ):
    
        """
        Data-dependent initialization, s.t. post-actnorm activations have
        zero mean and unitvariance.
        
        """
        
        if inputs.dim() == 4:
            inputs = einops.rearrange(inputs, "b c h w -> (b h w) c")

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)
