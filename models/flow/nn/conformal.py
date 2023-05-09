from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FlowTransform


class ConformalActNorm(FlowTransform):

    """
    Flow component of the type $x |-> a*x + b$, where a is a *scalar*.
    This is a conformal version of ActNorm, wherein a can be a vector.

    """

    def __init__(
        self,
        features: int,
    ) -> None:

        super().__init__()

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(features))

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4]
        shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self.log_scale.exp().view(shape), self.shift.view(shape)
        z = scale * x + shift

        if x.dim() == 4:
            B, C, H, W = x.size()
            logabsdet = x.new_zeros(B) + self.log_scale.sum() * C * H * W
        else:
            B, C = x.size()
            logabsdet = x.new_zeros(B) + self.log_scale.sum() * C

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
            B, C, H, W = z.size()
            logabsdet = z.new_zeros(B) - self.log_scale.sum() * C * H * W
        else:
            B, C = z.size()
            logabsdet = z.new_zeros(B) - self.log_scale.sum() * C

        return x, logabsdet

    def _initialize(
        self,
        inputs: torch.Tensor,
    ) -> None:

        """
        Data-dependent initialization, s.t. post-actnorm activations have
        zero mean and unit variance.

        """

        if inputs.dim() == 4:
            inputs = einops.rearrange(inputs, "b c h w -> (b h w) c")

        with torch.no_grad():
            std = inputs.std(dim=0).mean()  # scalar
            mu = (inputs / std).mean(dim=0)  # vactor
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class ConformalConv2D(FlowTransform):

    """
    1x1 convolution with a householder matrix kernel
    (orthogonal & symmetric, hence involutory)

    (*CHW) <-> [conv] <-> (z)

    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:

        super().__init__()

        # channel count
        self.num_channels = num_channels
        # identity matrix
        self.register_buffer("i", torch.eye(self.num_channels))
        # householder vector
        self.v = nn.Parameter(torch.randn(self.num_channels, 1))
        # bias term
        self.b = nn.Parameter(torch.zeros(self.num_channels))

    @property
    def filter(
        self,
    ) -> torch.Tensor:
        k = self.i - 2 * self.v @ self.v.T / torch.sum(self.v**2)
        return einops.rearrange(k, 'C c -> C c 1 1')

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.size()
        assert C == self.num_channels
        assert x.dim() == 4
        shape = (1, -1, 1, 1)

        z = F.conv2d(input=x, weight=self.filter)
        z = z + self.b.view(shape)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = z.size()
        assert C == self.num_channels
        assert z.dim() == 4
        shape = (1, -1, 1, 1)

        z = z - self.b.view(shape)
        x = F.conv_transpose2d(input=z, weight=self.filter)
        logabsdet = -z.new_zeros(B)

        return x, logabsdet


class PiecewiseConformalConv2D(FlowTransform):
    """
    Flow component of the type z = ||x|| >= 1? conv1(x): conv2(x)
    Both convs here are orthogonal and hence norm-preserving.
    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:

        super().__init__()

        self.num_channels = num_channels
        self.conv1 = ConformalConv2D(num_channels)
        self.conv2 = ConformalConv2D(num_channels)

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.size()
        assert C == self.num_channels

        # compute norm
        norm = einops.reduce(x**2, "B C H W -> B 1 H W", reduction="sum")
        # compute condition
        cond = einops.repeat(norm >= 1, 'B 1 H W -> B (C 1) H W', C=C)
        # compute output
        z1 = self.conv1(x, forward=True)[0]
        z2 = self.conv2(x, forward=True)[0]
        outputs = torch.where(cond, z1, z2)

        logabsdet = x.new_zeros(B)

        return outputs, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = z.size()
        assert C == self.num_channels

        # compute norm
        norm = einops.reduce(z**2, "B C H W -> B 1 H W", reduction="sum")
        # compute condition
        cond = einops.repeat(norm >= 1, 'B 1 H W -> B (C 1) H W', C=C)
        # compute output
        x1 = self.conv1(z, forward=False)[0]
        x2 = self.conv2(z, forward=False)[0]
        outputs = torch.where(cond, x1, x2)

        logabsdet = z.new_zeros(B)

        return outputs, logabsdet
