from typing import Tuple

import torch
import torch.nn
import torch.nn.functional


class CouplingNetwork(torch.nn.Module):

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def __init__(self, in_channels: int, hidden_channels: int, num_hidden: int, out_channels: int, num_params: int = 2):
    """

    :param in_channels: Input shape
    :param out_channels: Output shape
    :param num_hidden: Number of hidden layers
    :param hidden_channels: Features in each hidden layer
    :param num_params: Parameters to return (default: 2)
    """
    super().__init__()
    self.num_params = num_params
    self.out_dim = out_channels
    self.input = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding='same')
    self.hidden = torch.nn.ModuleList([torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same') for _ in range(num_hidden)])
    self.output = torch.nn.Conv2d(hidden_channels, out_channels * num_params, kernel_size=3, padding='same')

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def forward(
    self,
    x: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    st = self.input(x)
    st = torch.nn.functional.leaky_relu(st)
    for layer in self.hidden:
      st = layer(st)
      st = torch.nn.functional.leaky_relu(st)
    st = self.output(st)
    (s, t) = torch.chunk(st, self.num_params, dim=1)
    return s, torch.squeeze(t)

  # --------------------------------------------------------------------------------------------------------------------------------------------------
