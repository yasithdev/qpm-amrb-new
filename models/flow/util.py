from typing import Tuple

import torch

from . import FlowTransform

# --------------------------------------------------------------------------------------------------------------------------------------------------


def log_p_z(z: torch.Tensor, z_dist: torch.distributions.Distribution) -> torch.Tensor:
    """
    Given a base distribution and a latent-space point, return the log probability of that point
    :param z: latent-space point
    :param z_dist: base distribution
    :return: log probability of the latent-space point
    """
    return z_dist.log_prob(z)


# --------------------------------------------------------------------------------------------------------------------------------------------------


def log_p_x(
    flow: FlowTransform,
    x: torch.Tensor,
    z_dist: torch.distributions.Distribution,
) -> torch.Tensor:
    """
    Given a base distribution and a data-space point, return the log probability of that point
    :param flow: Normalizing flow from Rd -> Rd
    :param x: data-space point
    :param z_dist: base distribution
    :return: log probability of the data-space point, given the flow
    """
    z, log_det = flow(x)
    return log_p_z(z, z_dist) + log_det


# --------------------------------------------------------------------------------------------------------------------------------------------------


def proj(
    z: torch.Tensor,
    emb_dimsims: int,
) -> torch.Tensor:
    """
    Project Density Tensor from Ambient Space into Manifold Space

    :param z: Density in Ambient Space
    :param emb_dimsims: Dimensions of Manifold Space
    :return: Projection of Density from Ambient Space into Manifold Space
    """
    zu = z[:, :emb_dimsims]
    return zu


# --------------------------------------------------------------------------------------------------------------------------------------------------


def pad(
    m: torch.Tensor,
    off_emb_dimsims: int,
) -> torch.Tensor:
    """
    Project Density Tensor from Manifold Space into Ambient Space

    :param m: Density in Manifold Space
    :param off_emb_dimsims: Dimensions of Off-Manifold Space
    :param chw: Tuple of [C, H, W]
    :return: Projection of Density from Manifold Space into Ambient Space
    """
    z = torch.nn.functional.pad(m, (0, 0, 0, 0, 0, off_emb_dimsims))
    return z


# --------------------------------------------------------------------------------------------------------------------------------------------------


def decode_mask(
    channel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert channel_mask.dim() == 1, "Mask must be a 1D tensor."
    assert channel_mask.numel() > 0, "Mask cannot be empty."

    indices = torch.arange(len(channel_mask))
    i_channels = indices[~channel_mask]
    t_channels = indices[channel_mask]

    return i_channels, t_channels

