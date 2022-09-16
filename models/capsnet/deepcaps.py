"""
Implementation of 'DeepCaps: Going Deeper with Capsule Networks'

"""

from typing import Tuple, Union

import einops
import torch

from .caps import squash
from .common import capsule_norm


def softmax_3d(
    x: torch.Tensor,
) -> torch.Tensor:
    exp_x = torch.exp(x)
    return exp_x / einops.reduce(exp_x, "B D d h w -> B D 1 1 1", "sum")


class ConvCaps3D(torch.nn.Module):
    """
    3D Convolution Capsule Layer with Dynamic Routing.

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
        in_capsules: Tuple[int, int],
        out_capsules: Tuple[int, int],
        kernel_size: Union[int, Tuple[int, int, int]] = (1, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: str = "valid",
        groups: int = 1,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_iters = routing_iters
        # (B, C, D, H, W) -> (B, c * d, D, h, w)
        self.conv = torch.nn.Conv3d(
            in_channels=in_capsules[0],
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

        # prediction tensor (B, c, D, d, h, w)
        u = einops.rearrange(
            self.conv(x),
            "B (c d) D h w -> B c D d h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
        )

        # intialize log prior (B, D, d, h, w)
        b = torch.zeros(
            size=(
                u.size(0),
                self.in_capsules[1],
                self.out_capsules[1],
                u.size(4),
                u.size(5),
            )
        )

        # dynamic routing
        for _ in range(self.routing_iters):
            # softmax of b across d,h,w dims
            c = softmax_3d(b)
            # compute s (B, c, d, h, w)
            s = torch.einsum("BDdhw,BcDdhw->Bcdhw", c, u)
            # update b
            b += torch.einsum("BcDdhw,Bcdhw->BDdhw", u, squash(s))

        return s


class MaskCaps(torch.nn.Module):
    """
    Layer to obtain the probability distribution and capsule features.
    (No Parameters)

    The features only contain values from the highest-norm capsule.

    Op: (B, C, D) -> [Class Distribution, Features]

    Class Distribution: (B, D)
    Features: (B, C)

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

        # Extract most-activated capsule output (B, C)
        (B, C, _) = x.size()
        extracted = x.gather(
            dim=2,
            index=dist.argmax(dim=1).view(B, 1, 1).repeat(1, C, 1),
        )
        # (B, C, 1)
        features = extracted.view(B, C)
        # (B, C)

        return dist, features
