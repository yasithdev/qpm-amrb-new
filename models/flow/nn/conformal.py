from typing import Optional, Tuple, Union

import einops
import geotorch
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
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.shift = nn.Parameter(torch.zeros(features))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self.log_scale.exp().view(view_shape), self.shift.view(
            view_shape
        )
        z = scale * x + shift

        if x.dim() == 4:
            B, C, H, W = x.size()
            logabsdet = x.new_ones(B) * self.log_scale.sum() * C * H * W
        else:
            B, C = x.size()
            logabsdet = x.new_ones(B) * self.log_scale.sum() * C

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if z.dim() == 4 else (1, -1)

        scale, shift = self.log_scale.exp().view(view_shape), self.shift.view(
            view_shape
        )
        x = (z - shift) / scale

        if z.dim() == 4:
            B, C, H, W = z.size()
            logabsdet = z.new_ones(B) * -self.log_scale.sum() * C * H * W
        else:
            B, C = z.size()
            logabsdet = z.new_ones(B) * -self.log_scale.sum() * C

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


class ConformalConv2D_1x1(FlowTransform):

    """
    1x1 convolution with an orthogonally parametrized kernel

    (*CHW) <-> [conv] <-> (*CHW)

    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:

        """
        Args:
            num_channels (int): Number of input/output channels (kept same for invertibility)

        """

        super().__init__()

        self.num_channels = num_channels
        w_size = (self.num_channels, self.num_channels)

        # orthogonal constrained kernel
        self._weight = nn.Parameter(torch.Tensor(*w_size))
        geotorch.orthogonal(self, "_weight")

    @property
    def weight(self) -> torch.Tensor:
        return einops.rearrange(self._weight, "cO cI -> cO cI 1 1")

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # convolution
        B = x.size(0)
        z = F.conv2d(input=x, weight=self.weight)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # transpose convolution
        B = z.size(0)
        x = F.conv_transpose2d(input=z, weight=self.weight)
        logabsdet = -z.new_zeros(B)

        return x, logabsdet


class ConformalConv2D_KxK(FlowTransform):

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
        self.stride = k if isinstance(k, Tuple) else (k, k)
        self.kH, self.kW = self.stride
        self.x_channels = x_channels
        self.z_channels = x_channels * self.kH * self.kW

        # identity matrix
        self.register_buffer("i", torch.eye(self.z_channels))

        # householder vector
        self.v = nn.Parameter(torch.randn(self.z_channels, 1))

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

        z = F.conv2d(input=x, weight=self.filter, stride=self.stride)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = z.size()

        x = F.conv_transpose2d(input=z, weight=self.filter, stride=self.stride)
        logabsdet = -z.new_zeros(B)

        return x, logabsdet
