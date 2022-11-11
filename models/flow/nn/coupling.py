from typing import Optional, Tuple

import torch
import torch.nn

from . import FlowTransform


class AffineCoupling(FlowTransform):
    """
    (p1) ---------------(p1)
             ↓     ↓
            (s)   (t)
             ↓     ↓
    (p2) ---[x]---[+]---(p2)
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
        p1, p2 = torch.chunk(x, 2, dim=1)
        st = self.coupling_network(p1)
        s, t = torch.chunk(st, 2, dim=1)
        p2 = p2 * torch.exp(s) + t
        z = torch.concat((p1, p2), dim=1)
        logabsdet = s.sum([1, 2, 3])

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # inverse coupling
        p1, p2 = torch.chunk(z, 2, dim=1)
        st = self.coupling_network(p1)
        s, t = torch.chunk(st, 2, dim=1)
        p2 = (p2 - t) * torch.exp(-s)
        x = torch.cat((p1, p2), dim=1)
        logabsdet = s.sum([1, 2, 3])

        return x, -logabsdet


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

        self.w.append(torch.nn.Parameter(torch.randn(c * p, c, kH, kW)))
        self.b.append(torch.nn.Parameter(torch.randn(c * p)))

        for _ in range(1, self.num_blocks):
            self.w.append(torch.nn.Parameter(torch.randn(c * p, c * p, kH, kW)))
            self.b.append(torch.nn.Parameter(torch.randn(c * p)))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        st = x
        for i in range(self.num_blocks):
            w, b = self.w[i], self.b[i]
            st = torch.nn.functional.conv2d(
                input=st,
                weight=w,
                bias=b,
                padding=1,
            )
            st = torch.tanh(st)
        return st
