# --------------------------------------------------------------------------------------------------------------------------------------------------
from typing import Tuple

import torch
from einops import rearrange

from .square_nf import SquareNormalizingFlow


def log_p_z(
  z: torch.Tensor,
  z_dist: torch.distributions.Distribution
) -> torch.Tensor:
  """
    Given a base distribution and a latent-space point, return the log probability of that point
    :param z: latent-space point
    :param z_dist: base distribution
    :return: log probability of the latent-space point
    """
  return z_dist.log_prob(z)


def log_p_x(
  flow: SquareNormalizingFlow,
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
  z, log_det = flow.forward(x)
  return log_p_z(z, z_dist) + log_det


def proj(
  z: torch.Tensor,
  manifold_dims: int,
) -> torch.Tensor:
  """
  Project Density Tensor from Ambient Space into Manifold Space

  :param z: Density in Ambient Space
  :param manifold_dims: Dimensions of Manifold Space
  :return: Projection of Density from Ambient Space into Manifold Space
  """
  z = rearrange(z, 'b c h w -> b (c h w)')
  zu = z[:, :manifold_dims]
  return zu


def pad(
  zu: torch.Tensor,
  off_manifold_dims: int,
  chw: Tuple[int, int, int],
) -> torch.Tensor:
  """
  Project Density Tensor from Manifold Space into Ambient Space

  :param zu: Density in Manifold Space
  :param off_manifold_dims: Dimensions of Off-Manifold Space
  :param chw: Tuple of [C, H, W]
  :return: Projection of Density from Manifold Space into Ambient Space
  """
  c, h, w = chw
  z_pad = torch.constant_pad_nd(zu, (0, off_manifold_dims))
  z = rearrange(z_pad, 'b (c h w) -> b c h w', c=c, h=h, w=w)
  return z
