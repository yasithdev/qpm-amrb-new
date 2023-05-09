import torch

from typing import Union, Type
from ..common import npad

T = Union[Type[torch.nn.Conv2d], Type[torch.nn.ConvTranspose2d]]


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
        conv: T = torch.nn.Conv2d,
        norm=torch.nn.GroupNorm,
        activation=torch.nn.functional.leaky_relu,
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
        self.conv1 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            self.Norm(4, self.hidden_channels),  # type: ignore
        )
        self.conv2 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                bias=False,
            ),
            self.Norm(4, self.hidden_channels),  # type: ignore
        )
        self.conv3 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.hidden_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            self.Norm(4, self.out_channels),  # type: ignore
        )

        # if stride != 1, match identity to x using 1x1 convolution
        if self.stride != 1:
            self.convI = torch.nn.Sequential(
                self.Conv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                self.Norm(4, self.out_channels),  # type: ignore
            )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # forward on x
        x = xI = input
        x = self.conv1(x)
        x = self.activation(x)
        ph = npad(x.size(2) - 2, k=3, s=self.stride)
        pw = npad(x.size(3) - 2, k=3, s=self.stride)
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)

        # forward on xI
        if self.stride != 1:
            xI = self.convI(xI)

        # add xI to value
        x = x + xI
        x = self.activation(x)

        return x
