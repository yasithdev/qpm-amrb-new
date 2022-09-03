from typing import Tuple

import torch
import torch.nn

from .flow_transform import FlowTransform


class AffineCoupling(FlowTransform):
    """
    (x1) ---------------(z1)
             ↓     ↓
            (s1) (t1)
             ↓     ↓
    (x2) ---[x]---[+]---(z2)
    """

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, coupling_network: torch.nn.Module) -> None:
        super().__init__()
        self.coupling_network = coupling_network

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # coupling
        x1, x2 = torch.chunk(x, 2, dim=1)
        s1: torch.Tensor
        t1: torch.Tensor
        s1, t1 = self.coupling_network(x1)
        z1 = x1
        z2 = x2 * torch.exp(s1) + t1
        z = torch.concat((z1, z2), dim=1)
        log_det = s1.sum([1, 2, 3])

        return z, log_det

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def inverse(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inverse coupling
        z1, z2 = torch.chunk(z, 2, dim=1)
        s1, t1 = self.coupling_network(z1)
        x1 = z1
        x2 = (z2 - t1) * torch.exp(-s1)
        x = torch.cat((x1, x2), dim=1)
        inv_log_det = -s1.sum([1, 2, 3])

        return x, inv_log_det

    # --------------------------------------------------------------------------------------------------------------------------------------------------
