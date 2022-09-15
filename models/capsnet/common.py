from typing import Tuple

import einops
import torch


def capsule_norm(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L2 norm of a capsule tensor.

    Op: (B, C, D, *) -> (B, 1, D, *)

    Args:
        x (torch.Tensor): input capsule tensor (B, C, D, *)

    Returns:
        torch.Tensor: output norm tensor (B, 1, D, *)
    """
    eps = torch.finfo(x.dtype).eps
    x_norm_sq = torch.sum(x**2, dim=1, keepdim=True)
    x_norm = torch.sqrt(x_norm_sq + eps)
    return x_norm


def conv_to_caps(
    x: torch.Tensor,
    out_capsules: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert convolutional input to capsule output

    Op: (B, C * D, H, W) -> (B, C, D, H, W)

    Args:
        x (torch.Tensor): input convolutional tensor (B, C * D, H, W)

    Returns:
        torch.Tensor: output capsule tensor (B, C, D, H, W)
    """
    (C, D) = out_capsules

    z = einops.rearrange(x, "B (C D) H W -> B C D H W", C=C, D=D)
    return z
