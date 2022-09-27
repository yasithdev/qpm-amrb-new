from typing import Tuple, Union

import einops
import torch

from .caps import squash


class ConvCapsDR(torch.nn.Module):
    """
    2D Convolution Capsule Layer with Dynamic Routing.

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
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.kernel_size = kernel_size
        self.routing_iters = routing_iters

        # weights (D*C, D*c*d, kH, kW)
        self.conv = torch.nn.Conv2d(
            in_channels=in_capsules[1] * in_capsules[0],
            out_channels=in_capsules[1] * out_capsules[1] * out_capsules[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_capsules[1],
            bias=False,
        )

        # log prior (1, D, d, 1, 1)
        from ..common import get_conv_out_shape
        self.prior = torch.zeros(
            1,
            in_capsules[1],
            out_capsules[1],
            1,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, C * D, H, W)
        x = einops.rearrange(
            x,
            "B C D H W -> B (D C) H W",
            C=self.in_capsules[0],
            D=self.in_capsules[1],
        )
        # prediction tensor (B, D, c, d, h, w)
        u = einops.rearrange(
            self.conv(x),
            "B (D c d) h w -> B D c d h w",
            D=self.in_capsules[1],
            d=self.out_capsules[1],
            c=self.out_capsules[0],
        )

        # dynamic routing
        b = self.prior
        for _ in range(self.routing_iters):
            # coupling coefficients c_{ij} (softmax of b across dim=d)
            c = torch.softmax(b, dim=2)
            # current capsule output s_j (B, d, c)
            s = torch.einsum("BDdhw,BDcdhw->Bcdhw", c, u)
            # squashed capsule output s_j (B, d, c)
            v = squash(s)
            # agreement a_{ij} (scalar product of u and v)
            a = torch.einsum("BDcdhw,Bcdhw->BDdhw", u, v)
            # update b
            b = b + a

        return s
