from typing import Tuple

import torch


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
        conv=torch.nn.Conv2d,
        norm=torch.nn.BatchNorm2d,
        activation=torch.nn.functional.relu,
    ) -> None:

        super().__init__()

        # class names for convolution and normalization
        self.Conv = conv
        self.Norm = norm
        self.activation = activation

        # properties
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.stride = stride

        # layers
        self.conv1 = self.Conv(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.norm1 = self.Norm(
            num_features=self.hidden_channels,
        )
        self.conv2 = self.Conv(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=self.stride,
            padding="same",
        )
        self.norm2 = self.Norm(
            num_features=self.hidden_channels,
        )
        self.conv3 = self.Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.norm3 = self.Norm(
            num_features=self.out_channels,
        )

        # if stride != 1, match identity to x using 1x1 convolution
        if self.stride != 1:
            self.convI = self.Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding="same",
            )
            self.normI = self.Norm(
                num_features=self.out_channels,
            )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # forward on x
        x = xI = input
        x = self.norm1(self.conv1(x))
        x = self.activation(x)
        x = self.norm2(self.conv2(x))
        x = self.activation(x)
        x = self.norm3(self.conv3(x))

        # forward on xI
        if self.stride != 1:
            xI = self.normI(self.convI(xI))

        # add xI to value
        x = x + xI
        x = self.activation(x)

        return x
