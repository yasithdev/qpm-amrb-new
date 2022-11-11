from typing import Optional, Tuple, Union

import einops
import geotorch
import torch

from . import FlowTransform
from .normalization import ActNorm


class ConformalActNorm(ActNorm):

    """
    Flow component of the type x |-> a*x + b, where a is a scalar.
    This is a conformal version of ActNorm, wherein a can be a vector.

    """

    def __init__(
        self,
        features: int,
        latent_dim: int,
    ) -> None:

        super().__init__(features)

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.shift = torch.nn.Parameter(torch.zeros(features))
        self.latent_dim = latent_dim

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self.scale.view(view_shape), self.shift.view(view_shape)
        z = scale * x + shift
        logabsdet = self.latent_dim * self.log_scale.sum() * z.new_ones(x.size(0))

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4]
        view_shape = (1, -1, 1, 1) if z.dim() == 4 else (1, -1)

        scale, shift = self.scale.view(view_shape), self.shift.view(view_shape)
        x = (z - shift) / scale
        logabsdet = self.latent_dim * self.log_scale.sum() * x.new_ones(z.size(0))

        return x, -logabsdet

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
            std = inputs.std(dim=0).mean()
            mu = (inputs / std).mean(dim=0)
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
        self.weight = torch.nn.Parameter(torch.Tensor(*w_size))
        geotorch.orthogonal(self, "weight")

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # convolution
        z = torch.nn.functional.conv2d(
            input=x, weight=einops.rearrange(self.weight, "cO cI -> cO cI 1 1")
        )

        logabsdet = x.new_zeros(x.size(0))

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # transpose convolution
        x = torch.nn.functional.conv_transpose2d(
            input=z, weight=einops.rearrange(self.weight, "cO cI -> cO cI 1 1")
        )

        logabsdet = z.new_zeros(z.size(0))

        return x, -logabsdet


class ConformalConv2D_KxK(FlowTransform):

    """
    KxK convolution with a householder matrix kernel (orthogonal, symmetric, involutory)

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
        self.kH, self.kW = k
        self.x_channels = x_channels
        self.z_channels = x_channels * self.kH * self.kW
        self.matrix_size = self.z_channels

        # identity matrix
        self.i = torch.nn.Parameter(torch.eye(self.matrix_size), requires_grad=False)

        # householder vector
        self.v = torch.nn.Parameter(torch.randn(self.matrix_size, 1))

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

        assert x.size(2) % self.kH == 0
        assert x.size(3) % self.kW == 0

        z = torch.nn.functional.conv2d(
            input=x,
            weight=self.filter,
            stride=(self.kH, self.kW),
        )

        logabsdet = x.new_zeros(x.size(0))

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = torch.nn.functional.conv_transpose2d(
            input=z,
            weight=self.filter,
            stride=(self.kH, self.kW),
        )

        logabsdet = z.new_zeros(x.size(0))

        return x, -logabsdet
