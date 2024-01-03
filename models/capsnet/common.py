from typing import Tuple

import einops
import torch


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


def flatten_caps(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten Capsule Layer.
    (No Parameters)

    Op: (B, C, ...) -> (B, C, [...])


    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count
    H/W: Input Height/Width

    """
    return x.flatten(start_dim=2)
