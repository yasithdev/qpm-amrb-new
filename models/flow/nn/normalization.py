from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

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
        self.eps = 1e-4

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
        z = (scale + self.eps) * x + shift

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
        x = (z - shift) / (scale + self.eps)

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

class Dequantize(FlowTransform):

    def __init__(
        self,
        alpha: float = 1e-5,
        quants: int = 256,
    ):
        """Dequantize

        Args:
            alpha: small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants: Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        logabsdet = x.new_zeros(B)
        z, logabsdet = self.dequant(x, logabsdet)
        z, logabsdet = self.sigmoid(z, logabsdet, reverse=True)
        return z, logabsdet
    
    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        logabsdet = z.new_zeros(B)
        z, logabsdet = self.sigmoid(z, logabsdet, reverse=False)
        z = z * self.quants
        logabsdet += np.log(self.quants) * np.prod(z.shape[1:])
        z = torch.floor(z).clamp(min=0, max=self.quants - 1).int()
        return z, logabsdet

    def sigmoid(
        self,
        z: torch.Tensor,
        logabsdet: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Applies an invertible sigmoid transformation
        if not reverse:
            logabsdet += (-z - 2 * F.softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            logabsdet += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            logabsdet += (-torch.log(z) - torch.log(1 - z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1 - z)
        return z, logabsdet

    def dequant(
        self,
        z: torch.Tensor,
        logabsdet: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform discrete values to continuous volumes
        z = z.float()
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        logabsdet -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, logabsdet
