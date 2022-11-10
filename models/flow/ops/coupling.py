from typing import Optional, Tuple

import torch
import torch.nn

from . import FlowTransform


class AffineCoupling(FlowTransform):
    """
    (x1) ---------------(z1)
             ↓     ↓
            (s1) (t1)
             ↓     ↓
    (x2) ---[x]---[+]---(z2)
    """

    def __init__(
        self,
        coupling_network: torch.nn.Module,
    ) -> None:

        super().__init__()

        self.coupling_network = coupling_network

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # coupling
        x1, x2 = torch.chunk(x, 2, dim=1)
        s1: torch.Tensor
        t1: torch.Tensor
        s1, t1 = self.coupling_network(x1)
        z1 = x1
        z2 = x2 * torch.exp(s1) + t1
        z = torch.concat((z1, z2), dim=1)
        logabsdet = s1.sum([1, 2, 3])

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inverse coupling
        z1, z2 = torch.chunk(z, 2, dim=1)
        s1, t1 = self.coupling_network(z1)
        x1 = z1
        x2 = (z2 - t1) * torch.exp(-s1)
        x = torch.cat((x1, x2), dim=1)
        inv_logabsdet = s1.sum([1, 2, 3])

        return x, -inv_logabsdet


class CouplingNetwork(torch.nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_blocks: int,
        num_params: int = 2,
    ):
        """

        :param num_channels: channel size of conv2d layers
        :param num_blocks: Number of conv2d blocks
        :param num_params: Parameters to return (default: 2)
        """
        super().__init__()

        assert num_blocks >= 1

        self.num_params = num_params
        self.num_blocks = num_blocks
        self.num_channels = num_channels

        self.w = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()

        kH = kW = 3
        c = num_channels
        p = num_params

        self.w.append(torch.randn(c*p, c, kH, kW))
        self.b.append(torch.randn(c*p))

        for _ in range(1, self.num_blocks):
            self.w.append(torch.randn(c*p, c*p, kH, kW))
            self.b.append(torch.randn(c*p))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        st = x
        for i in range(self.num_blocks):
            w, b = self.w[i], self.b[i]
            st = torch.nn.functional.conv2d(
                input=st,
                weight=w,
                bias=b,
                padding=1,
            )
            st = torch.nn.functional.leaky_relu(st)
        
        (s, t) = torch.chunk(st, self.num_params, dim=1)
        return s, torch.squeeze(t)
