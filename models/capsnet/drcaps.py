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
        k = kernel_size
        k = (1, k, k) if isinstance(k, int) else (1, *k)
        s = stride
        self.stride = (1, s, s) if isinstance(s, int) else (1, *s)
        self.padding = padding
        self.routing_iters = routing_iters
        C, D = in_capsules
        c, d = out_capsules
        # convolution weights and biases
        self.conv_k = torch.nn.Parameter(torch.randn(c * d, C, *k))
        self.conv_b = torch.nn.Parameter(torch.randn(c * d))
        # log prior
        self.prior = torch.nn.Parameter(torch.zeros(1, d, D, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # prediction tensor (B, C, D, H, W) -> (B, c*d, D, h, w)
        u = torch.nn.functional.conv3d(
            input=x,
            weight=self.conv_k,
            bias=self.conv_b,
            stride=self.stride,
            padding=self.padding,
        )
        # rearrange prediction tensor
        u = einops.rearrange(
            u,
            "B (c d) D h w -> B c d D h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
            D=self.in_capsules[1],
        )
        # dynamic routing
        b = self.prior
        for _ in range(self.routing_iters):
            # coupling coefficients c_{ij} (softmax of b across dim=d)
            c = torch.softmax(b, dim=1)
            # current capsule output s_j (B, c, d, h, w)
            s = torch.einsum("BdDhw,BcdDhw->Bcdhw", c, u)
            # squashed capsule output s_j (B, c, d, h, w)
            v = squash(s)
            # agreement a_{ij} (scalar product of u and v)
            a = torch.einsum("BcdDhw,Bcdhw->BdDhw", u, v)
            # update b
            b = b + a
        # post-routing
        c = torch.softmax(b, dim=1)
        s = torch.einsum("BdDhw,BcdDhw->Bcdhw", c, u)
        return s
