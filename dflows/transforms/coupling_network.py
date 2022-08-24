from typing import Tuple

import torch
import torch.nn
import torch.nn.functional


class CouplingNetwork(torch.nn.Module):

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        num_channels: int,
        num_hidden_channels: int,
        num_hidden_layers: int,
        num_params: int = 2,
    ):
        """

        :param num_channels: Channel size of input / output layers
        :param num_hidden_channels: Channel size of each hidden layer
        :param num_hidden_layers: Number of hidden layers
        :param num_params: Parameters to return (default: 2)
        """
        super().__init__()
        self.num_params = num_params
        self.input = torch.nn.Conv2d(
            num_channels, num_hidden_channels, kernel_size=3, padding="same"
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
