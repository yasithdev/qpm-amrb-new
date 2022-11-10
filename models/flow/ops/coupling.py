from typing import Tuple

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
        num_hidden_channels: int,
        num_hidden_layers: int,
        num_params: int = 2,
    ):
        """

        :param num_channels: Channel size of input / output conv2d layers
        :param num_hidden_channels: Channel size of hidden conv2d layers
        :param num_hidden_layers: Number of hidden conv2d layers
        :param num_params: Parameters to return (default: 2)
        """
        super().__init__()

        self.num_params = num_params
        self.input = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            padding="same",
        )
        self.hidden = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    num_hidden_channels,
                    num_hidden_channels,
                    kernel_size=3,
                    padding="same",
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output = torch.nn.Conv2d(
            num_hidden_channels,
            num_channels * num_params,
            kernel_size=3,
            padding="same",
        )

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
