# --------------------------------------------------------------------------------------------------------------------------------------------------
import torch

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


def set_requires_grad(
  module: torch.nn.Module,
  flag: bool
):
  """
  Sets requires_grad flag of all parameters of a torch.nn.module
  :param module: torch.nn.module
  :param flag: Flag to set requires_grad to
  """

  for param in module.parameters():
    param.requires_grad = flag
