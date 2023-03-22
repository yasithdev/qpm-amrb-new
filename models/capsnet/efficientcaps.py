"""
Implementation of 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'

"""

from typing import Tuple
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
    # return v = s * (1-e^-||s||)/||s||
    return s * ((1 - torch.exp(-s_norm)) / s_norm)


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
        # log priors (d, D)
        self.prior = torch.nn.Parameter(torch.randn(d, D))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # parameters
        B = x.size(0)
        # prediction tensor u_{j|i} (B, c, d, D)
        u = squash(torch.einsum("BCD,cdCD->BcdD", x, self.weight))
        # prior routing weights (B, d, D)
        b = self.prior.tile(B, 1, 1)
        # updated routing weights (B, d, D)
        b = b + (torch.einsum("BcdD,BcdD->BdD", u, u) / self.nc)
        # weighted capsule prediction (B, c, d)
        v = torch.einsum("BcdD,BdD->Bcd", u, b.softmax(dim=1))
        return v
