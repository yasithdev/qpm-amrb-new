from typing import Tuple

import torch.nn


class FlowTransform(torch.nn.Module):

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def forward(
    self,
    x: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    transform a tensor towards latent space (forward mode)

    :param x: Tensor
    :return: Tuple [Transformed Tensor, Determinant of Jacobian]

    """
    raise NotImplementedError()

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def inverse(
    self,
    z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    transform a tensor towards data space (inverse mode)

    :param z: Tensor
    :return: Tuple [Transformed Tensor, Determinant of Jacobian]
    """
    raise NotImplementedError()

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def jacobian(
    self,
    z: torch.Tensor
  ) -> torch.Tensor:
    raise NotImplementedError()
