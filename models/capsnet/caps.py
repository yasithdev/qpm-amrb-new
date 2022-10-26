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
    ) -> None:
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        k = kernel_size
        k = (k, k) if isinstance(k, int) else k
        s = stride
        self.stride = (s, s) if isinstance(s, int) else s
        self.padding = padding
        C, D = in_capsules
        c, d = out_capsules
        self.groups = D
        self.conv_k = torch.nn.Parameter(torch.randn(d * c, C, *k))
        self.conv_b = torch.nn.Parameter(torch.randn(d * c))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D*C, H, W)
        s = einops.rearrange(
            x,
            "B C D H W -> B (D C) H W",
            C=self.in_capsules[0],
            D=self.in_capsules[1],
        )
        # (B, D*C, H, W) -> (B, d*c, h, w)
        s = torch.nn.functional.conv2d(
            input=s,
            weight=self.conv_k,
            bias=self.conv_b,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        # (B, d*c, h, w) -> (B, c, d, h, w)
        s = einops.rearrange(
            s,
            "B (d c) h w -> B c d h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
        )
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


class LinearCapsDR(torch.nn.Module):
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
        C, D = in_capsules
        c, d = out_capsules
        # weights (d, D, c, C)
        self.weight = torch.nn.Parameter(torch.randn(d, D, c, C))
        # log prior (1, d, D)
        self.prior = torch.nn.Parameter(torch.zeros(1, d, D))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = einops.rearrange(x, "B C D -> B D C")
        # prediction tensor u_{j|i} (B, d, D, c)
        u = torch.einsum("BDC,dDcC->BdDc", x, self.weight)
        # dynamic routing
        b = self.prior
        for _ in range(self.routing_iters):
            # coupling coefficients c_{ij} (softmax of b across dim=d)
            c = torch.softmax(b, dim=1)
            # current capsule output s_j (B, d, c)
            s = torch.einsum("BdD,BdDc->Bdc", c, u)
            # squashed capsule output s_j (B, d, c)
            v = squash(s)
            # agreement a_{ij} (scalar product of u and v)
            a = torch.einsum("BdDc,Bdc->BdD", u, v)
            # update b
            b = b + a
        # post-routing
        c = torch.softmax(b, dim=1)
        s = torch.einsum("BdD,BdDc->Bcd", c, u)
        return s


class MaskCaps(torch.nn.Module):
    """
    Layer to obtain the capsule logits and latent.
    (No Parameters)

    The latent contain features from all capsules.
    However, only the highest-norm capsule will have non-zero features.

    Op: (B, C, D) -> [logits, latent]

    logits: (B, D)
    latent: (B, C * D)

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

        # Compute logits via capsule norm (B, D)
        logits = capsule_norm(x).squeeze()

        # Mask most-activated capsule output (B, C * D)
        (B, C, D) = x.size()
        I = torch.eye(D, device=x.device)
        mask = I.index_select(
            dim=0,
            index=logits.argmax(dim=1),
        )
        # (B, D)
        latent = (x * mask.reshape(B, 1, D)).reshape(B, C * D)
        # (B, C * D)

        return logits, latent
