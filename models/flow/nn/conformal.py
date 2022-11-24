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

    def forward(
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

    def inverse(
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
    KxK convolution with a householder matrix kernel
    (orthogonal & symmetric, hence involutory)

    (*CHW) <-> [conv] <-> (z)

    """

    def __init__(
        self,
        x_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
    ) -> None:

        super().__init__()

        k = kernel_size
        k = k if isinstance(k, Tuple) else (k, k)
        self.stride = k
        self.kH, self.kW = k
        self.x_channels = x_channels
        self.z_channels = x_channels * self.kH * self.kW

        # identity matrix
        self.register_buffer("i", torch.eye(self.z_channels))

        # householder vector
        self.v = nn.Parameter(torch.randn(self.z_channels, 1))

        # bias term
        self.b = nn.Parameter(torch.zeros(self.z_channels))

    @property
    def filter(
        self,
    ) -> torch.Tensor:

        k = self.i - 2 * self.v @ self.v.T / torch.sum(self.v**2)
        return einops.rearrange(
            k,
            "z (x kH kW) -> z x kH kW",
            x=self.x_channels,
            kH=self.kH,
            kW=self.kW,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.size()
        assert H % self.kH == 0
        assert W % self.kW == 0
        assert x.dim() == 4
        shape = (1, -1, 1, 1)

        z = F.conv2d(input=x, weight=self.filter, stride=self.stride)
        z = z + self.b.view(shape)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = z.size()
        assert z.dim() == 4
        shape = (1, -1, 1, 1)

        z = z - self.b.view(shape)
        x = F.conv_transpose2d(input=z, weight=self.filter, stride=self.stride)
        logabsdet = -z.new_zeros(B)

        return x, logabsdet
