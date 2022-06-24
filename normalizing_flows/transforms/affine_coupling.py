from typing import Tuple

import torch
import torch.nn

from .coupling_network import CouplingNetwork
from .flow_transform import FlowTransform


class AffineCouplingTransform(FlowTransform):
  """
  (x1) ---------------(h1)(h2)---------------(z2)(z1)
           ↓     ↓       x        ↓     ↓        x
          (s1) (t1)    [swap]    (s2) (t2)    [swap]
           ↓     ↓       x        ↓     ↓        x
  (x2) ---[x]---[+]---(h2)(h1)---[x]---[+]---(z1)(z2)

  """

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def __init__(
    self,
    ct1: torch.nn.Module,
    ct2: torch.nn.Module,
  ) -> None:
    super().__init__()
    self.ct1 = ct1
    self.ct2 = ct2

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def forward(
    self,
    x: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    log_det = torch.zeros(x.size(0), device=x.device)
    # split
    x1, x2 = torch.chunk(x, 2, dim=-1)
    # coupling
    s1: torch.Tensor
    t1: torch.Tensor
    s1, t1 = self.ct1(x1)
    log_det += s1.sum([-2, -1])
    h1 = x1
    h2 = x2 * torch.exp(s1) + t1
    # coupling
    s2: torch.Tensor
    t2: torch.Tensor
    s2, t2 = self.ct2(h2)
    log_det += s2.sum([-2, -1])
    z2 = h2
    z1 = h1 * torch.exp(s2) + t2
    # concatenate
    z = torch.cat((z1, z2), dim=-1)
    return z, log_det

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def inverse(
    self,
    z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_log_det = torch.zeros(z.size(0), device=z.device)
    # split
    z1, z2 = torch.chunk(z, 2, dim=-1)
    # inverse coupling
    s2, t2 = self.ct2(z2)
    inv_log_det += s2.sum([-2, -1])
    h2 = z2
    h1 = (z1 - t2) * torch.exp(-s2)
    # inverse coupling
    s1, t1 = self.ct1(h1)
    inv_log_det += s1.sum([-2, -1])
    x1 = h1
    x2 = (h2 - t1) * torch.exp(-s1)
    # concatenate
    x = torch.cat((x1, x2), dim=-1)
    return x, -inv_log_det

  # --------------------------------------------------------------------------------------------------------------------------------------------------

  def jacobian(
    self,
    z: torch.Tensor,
  ) -> torch.Tensor:
    pass

