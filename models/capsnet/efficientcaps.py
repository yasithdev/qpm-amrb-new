"""
Implementation of 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'

"""

from typing import Tuple

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


class LinearCaps(torch.nn.Module):
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

        # weights (c, C, D, d)
        self.w = torch.nn.Parameter(
            torch.randn(
                out_capsules[0],
                in_capsules[0],
                in_capsules[1],
                out_capsules[1],
            )
        )

        # log priors (D, d)
        self.b = torch.nn.Parameter(
            torch.randn(
                in_capsules[1],
                out_capsules[1],
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # capsule prediction (B, c, D, d)
        u = torch.einsum("BCD,cCDd->BcDd", x, self.w)
        # scaled dot product attention of u (B, D, d)
        c = torch.softmax(
            torch.einsum("BcDd,BcDd->BDd", u, u) / self.nc,
            dim=2,
        )
        # weighted capsule prediction (B, c, d)
        s = torch.einsum("BDd,BcDd->Bcd", c + self.b, u)

        return s
