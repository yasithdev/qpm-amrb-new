import torch


def capsule_norm(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L2 norm of a capsule tensor.
    (B, C, D, *) -> (B, 1, D, *)

    Args:
        x (torch.Tensor): input capsule tensor (B, C, D, *)

    Returns:
        torch.Tensor: output norm tensor (B, 1, D, *)
    """
    eps = torch.finfo(x.dtype).eps
    x_norm_sq = torch.sum(x**2, dim=1, keepdim=True)
    x_norm = torch.sqrt(x_norm_sq + eps)
    return x_norm
