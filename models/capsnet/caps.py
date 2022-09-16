"""
Implementation of 'Dynamic Routing Between Capsules'

"""

from typing import Tuple, Union

import einops
import torch

from .common import capsule_norm


def squash(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Activation function to normalize capsules into unit vectors

    Args:
        x (torch.Tensor): input tensor (B, C, D, *)

    Returns:
        torch.Tensor: output tensor (B, C, D, *)
    """
    x_norm = capsule_norm(x)
    a = (x_norm**2) / (1 + x_norm**2)
    b = x / x_norm
    return a * b


class ConvCaps2D(torch.nn.Module):
    """
    2D Convolution Capsule Layer.

    Op: (B, C, D, H, W) -> (B, c, d, h, w)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count
    H/W: Input Height/Width

    c: Output capsule channels
    d: Output capsule count
    h/w: Output Height/Width

    """

    def __init__(
        self,
        in_capsules: Tuple[int, int],  # (C, D)
        out_capsules: Tuple[int, int],  # (c, d)
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.conv = torch.nn.Conv2d(
            in_channels=in_capsules[0] * in_capsules[1],
            out_channels=out_capsules[0] * out_capsules[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # (B, C, D, H, W)
        s = einops.rearrange(
            x,
            "B C D H W -> B (C D) H W",
            C=self.in_capsules[0],
            D=self.in_capsules[1],
        )
        # (B, C * D, H, W)
        s = self.conv(s)
        # (B, c * d, h, w)
        s = einops.rearrange(
            s,
            "B (c d) h w -> B c d h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
        )
        # (B, c, d, h, w)
        return s


class FlattenCaps(torch.nn.Module):
    """
    Flatten Capsule Layer.
    (No Parameters)

    Op: (B, C, D, H, W) -> (B, C, D * H * W)


    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count
    H/W: Input Height/Width

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        s = einops.rearrange(x, "B C D H W -> B C (D H W)")
        return s


class LinearCaps(torch.nn.Module):
    """
    Linear Capsule Layer with Dynamic Routing.

    Op: (B, C, D) -> (B, c, d)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count

    c: Output capsule channels
    d: Output capsule count

    """

    def __init__(
        self,
        in_capsules: Tuple[int, int],  # (C, D)
        out_capsules: Tuple[int, int],  # (c, d)
        routing_iters: int = 3,
    ):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_iters = routing_iters

        # weights (d, D, c, C)
        self.weight = torch.nn.Parameter(
            torch.randn(
                out_capsules[1],
                in_capsules[1],
                out_capsules[0],
                in_capsules[0],
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # prediction tensor (B, d, D, c)
        x = einops.rearrange(x, 'B C D -> B D C')
        u = torch.einsum("BDC,dDcC->BdDc", x, self.weight)

        # intialize log prior (B, d, D)
        b = torch.zeros(
            size=(
                u.size(0),
                self.out_capsules[1],
                self.in_capsules[1],
            )
        )

        # dynamic routing
        for _ in range(self.routing_iters):
            # softmax of b across dim=d
            c = torch.softmax(b, dim=1)
            # compute s (B, d, c)
            s = torch.einsum("BdD,BdDc->Bdc", c, u)
            # update b
            b += torch.einsum("BdDc,Bdc->BdD", u, squash(s))

        return einops.rearrange(s, 'B d c -> B c d')


class MaskCaps(torch.nn.Module):
    """
    Layer to obtain the probability distribution and capsule features.
    (No Parameters)

    The features contain values from all capsules.
    However, only the highest-norm capsule will have non-zero values.

    Op: (B, C, D) -> [Class Distribution, Features]

    Class Distribution: (B, D)
    Features: (B, C * D)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute class distribution (B, D)
        x_norm = capsule_norm(x).squeeze()
        dist = torch.nn.functional.softmax(x_norm, dim=1)

        # Mask most-activated capsule output (B, C * D)
        (B, C, D) = x.size()
        mask = torch.eye(D).index_select(
            dim=0,
            index=dist.argmax(dim=1),
        )
        # (B, D)
        features = (x * mask.reshape(B, 1, D)).reshape(B, C * D)
        # (B, C * D)

        return dist, features
