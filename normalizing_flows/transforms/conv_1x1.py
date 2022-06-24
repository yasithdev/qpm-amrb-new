from typing import Tuple

import numpy as np
import torch
import torch.nn
import torch.nn.functional

from .flow_transform import FlowTransform


class Conv1x1(FlowTransform):
  """
  (x) <-> [conv] <-> (z)
  """

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def __init__(self, num_channels: int):
    """

    :param num_channels: Number of input/output channels (should be same to be invertible)
    """
    super().__init__()
    self.num_channels = num_channels
    w_init = np.random.randn(self.num_channels, self.num_channels)
    w_init = np.linalg.qr(w_init)[0].astype(np.float32)
    self.weight = torch.nn.Parameter(torch.from_numpy(w_init))

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def forward(
    self,
    x: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # convolution
    ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)
    weight = self.weight
    weight = weight.view(self.num_channels, self.num_channels, 1, 1)
    z = torch.nn.functional.conv2d(x, weight)

    return z, ldj

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def inverse(
    self,
    z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # inverse convolution
    ldj = torch.slogdet(self.weight)[1] * z.size(2) * z.size(3)
    weight = torch.inverse(self.weight.double()).float()
    weight = weight.view(self.num_channels, self.num_channels, 1, 1)
    x = torch.nn.functional.conv2d(z, weight)

    return x, -ldj
