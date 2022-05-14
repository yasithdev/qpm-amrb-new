from typing import Tuple

import torch
import torch.nn
import torch.nn.functional


class CouplingNetwork(torch.nn.Module):

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def __init__(self, in_dim: int, hidden_dim: int, num_hidden: int, out_dim: int, num_params: int = 2):
    """

    :param in_dim: Input shape
    :param out_dim: Output shape
    :param num_hidden: Number of hidden layers
    :param hidden_dim: Features in each hidden layer
    :param num_params: Parameters to return (default: 2)
    """
    super().__init__()
    self.num_params = num_params
    self.out_dim = out_dim
    self.input = torch.nn.Linear(in_dim, hidden_dim)
    self.hidden = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
    self.output = torch.nn.Linear(hidden_dim, out_dim * num_params)

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def forward(
    self,
    x: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    st = torch.nn.functional.leaky_relu(self.input(x))
    for layer in self.hidden:
      st = torch.nn.functional.leaky_relu(layer(st))
    st = self.output(st)
    (s, t) = torch.chunk(st, self.num_params, dim=-1)
    return s, torch.squeeze(t)
