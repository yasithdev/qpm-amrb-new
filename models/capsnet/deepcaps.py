"""
Implementation of 'DeepCaps: Going Deeper with Capsule Networks'

"""

from typing import Tuple, Union

import einops
import torch

from .caps import squash


def softmax_3d(
    x: torch.Tensor,
) -> torch.Tensor:
    exp_x = torch.exp(x)
    return exp_x / einops.reduce(exp_x, "B d D h w -> B 1 D 1 1", "sum")


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
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",
        routing_iters: int = 3,
    ):
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
        # initial prediction tensor (B, C, D, H, W) -> (B, c*d, D, h, w)
        u = torch.nn.functional.conv3d(
            input=x,
            weight=self.conv_k,
            bias=self.conv_b,
            stride=self.stride,
            padding=self.padding,
        )
        # reshaped prediction tensor
        u = einops.rearrange(
            u,
            "B (c d) D h w -> B c d D h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
        )
        # dynamic routing
        b = self.prior
        for _ in range(self.routing_iters):
            # softmax of b across d,h,w dims (B, d, D, h, w)
            c = torch.nn.functional.softmax(b, dim=1)
            # compute capsule output (B, c, d, h, w)
            s = torch.einsum("BdDhw,BcdDhw->Bcdhw", c, u)
            # squash capsule output s_j (B, d, c)
            v = squash(s)
            # agreement a_{ij} (scalar product of u and v)
            a = torch.einsum("BcdDhw,Bcdhw->BdDhw", u, v)
            # update b
            b = b + a
        # post-routing
        c = torch.nn.functional.softmax(b, dim=1)
        s = torch.einsum("BdDhw,BcdDhw->Bcdhw", c, u)
        return s


class MaskCaps(torch.nn.Module):
    """
    Layer to obtain the capsule logits and latent.
    (No Parameters)

    The latent only contain features from the highest-norm capsule.

    Op: (B, C, D) -> [logits, latent]

    logits: (B, D)
    latent: (B, C)

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
        (B, C, _) = x.size()
        # Compute logits -> (B, D)
        logits = torch.norm(x, p=2, dim=1)
        # Generate index to extract capsule (B, C, 1)
        index = torch.argmax(logits, dim=1).view(B, 1, 1).repeat(1, C, 1)
        # Extract capsule -> (B, C, 1)
        extracted = torch.gather(x, dim=2, index=index)
        # (B, C, 1) -> (B, C)
        latent = extracted.view(B, C)
        return logits, latent
