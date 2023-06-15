"""
Implementation of 'Dynamic Routing Between Capsules'

"""

from typing import Tuple, Union

import einops
import torch


def squash(
    s: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Activation function to squash vectors into magnitude [0,1]

    Args:
        x (torch.Tensor): input tensor (B, C, D, *)
        dim (int): axis to squash

    Returns:
        torch.Tensor: output tensor (B, C, D, *)
    """
    # compute ||s|| and clamp it
    ε = torch.finfo(s.dtype).eps
    s_norm = torch.norm(s, p=2, dim=dim, keepdim=True)
    s_norm = torch.clamp(s_norm, min=ε)
    # return v = s * tanh(||s||)/||s||
    return s * (torch.tanh(s_norm) / s_norm)


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

    Op: (B, C, i) -> (B, c, j)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    i: Input capsule count

    c: Output capsule channels
    j: Output capsule count

    """

    def __init__(
        self,
        in_capsules: Tuple[int, int],  # (C, i)
        out_capsules: Tuple[int, int],  # (c, j)
        routing_iters: int = 3,
    ):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_iters = routing_iters
        C, i = in_capsules
        c, j = out_capsules
        # weights (j, i, c, C)
        self.weight = torch.nn.Parameter(torch.randn(j, i, c, C))
        self.nc = c ** 0.5

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B = x.size(0)
        # input tensor (BCi...) -> (B...iC)
        x = einops.rearrange(x, 'B C i ... -> B ... i C')
        # prediction tensor u_{j|i} (...jic)
        u = torch.einsum("...iC,jicC->...jic", x, self.weight)
        # coupling logits (...ji)
        b = x.new_zeros(*u.size()[-3:-1])
        # dynamic routing
        for _ in range(self.routing_iters):
            c = b.softmax(dim=-2)
            # initial capsule output (...jc)
            s = torch.einsum("...jic,...ji->...jc", u, c)
            v = squash(s, dim=-2)
            # add agreement to coupling logits: dot product of v with each u (...ji)
            b = b + torch.einsum("...jc,...jic->...ji", v, u)
        # output tensor
        v = einops.rearrange(v, "B ... j c -> B c j ...")
        return v


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
        logits = x.norm(p=2, dim=1)

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
