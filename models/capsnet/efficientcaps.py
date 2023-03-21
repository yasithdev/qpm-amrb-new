"""
Implementation of 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'

"""

from typing import Tuple

import einops
import torch


def squash(
    x: torch.Tensor,
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
    # nonlinear scaling (B, 1, D)
    s = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    s = 1 - torch.exp(-s)
    # unit vectorization (B, C, D)
    u = torch.nn.functional.normalize(x, p=2, dim=dim)
    # scaled unit vector
    return s * u


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
        self.nc = out_capsules[0] ** 0.5
        C, D = in_capsules
        c, d = out_capsules
        # weights (c, d, C, D)
        self.weight = torch.nn.Parameter(torch.randn(c, d, C, D))
        # log priors (1, d, D)
        self.prior = torch.nn.Parameter(torch.zeros(1, d, D))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # prediction tensor u_{j|i} (B, c, d, D)
        u = torch.einsum("BCD,cdCD->BcdD", x, self.weight)
        # agreement: prior + self-attention of u (B, d, D)
        b = self.prior + (torch.einsum("BcdD,BcdD->BdD", u, u) / self.nc)
        # weighted capsule prediction (B, c, d)
        s = torch.einsum("BdD,BcdD->Bcd", b.softmax(dim=1), u)
        return s
