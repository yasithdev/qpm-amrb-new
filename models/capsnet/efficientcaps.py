"""
Implementation of 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'

"""

from typing import Tuple

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
    a = 1 - 1 / torch.exp(x_norm)
    b = x / x_norm
    return a * b


class LinearCapsAR(torch.nn.Module):
    """
    Linear Capsule Layer with Attention Routing.

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
    ) -> None:

        super().__init__()

        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.nc = out_capsules[0] ** 0.5

        # weights (d, D, c, C)
        self.weight = torch.nn.Parameter(
            torch.randn(
                out_capsules[1],
                in_capsules[1],
                out_capsules[0],
                in_capsules[0],
            ),
        )

        # log priors (1, d, D)
        self.prior = torch.nn.Parameter(
            torch.randn(
                1,
                out_capsules[1],
                in_capsules[1],
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = einops.rearrange(x, "B C D -> B D C")
        # prediction tensor u_{j|i} (B, d, D, c)
        u = torch.einsum("BDC,dDcC->BdDc", x, self.weight)
        
        # scaled dot product attention of u (B, d, D)
        c = torch.softmax(
            torch.einsum("BdDc,BdDc->BdD", u, u) / self.nc,
            dim=2,
        )
        # weighted capsule prediction (B, d, c)
        s = torch.einsum("BdD,BdDc->Bdc", c + self.prior, u)

        return einops.rearrange(s, "B d c -> B c d")
