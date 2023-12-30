import torch
import torch.nn.functional as F

from typing import Union, Type

T = Union[Type[torch.nn.Conv2d], Type[torch.nn.ConvTranspose2d]]


def npad(
    d: int,
    k: int,
    s: int,
) -> int:
    return (s - (d - k - 1) % s) % s


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
        conv: T = torch.nn.Conv2d,
        norm=torch.nn.BatchNorm2d,
        activation=F.leaky_relu,
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
        assert self.stride in [1, 2]

        # 1x1 convolution layer
        self.conv1 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            self.Norm(self.hidden_channels),
        )

        args = {}
        if self.stride == 2 and self.Conv == torch.nn.ConvTranspose2d:
            args["output_padding"] = 1
        # 3x3 convolution layer
        self.conv2 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False,
                **args,
            ),
            self.Norm(self.hidden_channels),
        )

        # 1x1 convolution layer
        self.conv3 = torch.nn.Sequential(
            self.Conv(
                in_channels=self.hidden_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            self.Norm(self.out_channels),
        )

        # if stride != 1, match identity to x using stridexstride convolution
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.convI = torch.nn.Sequential(
                self.Conv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.stride,
                    stride=self.stride,
                    bias=False,
                ),
                self.Norm(self.out_channels),
            )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # forward on x
        x = xI = input
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)

        # forward on xI
        if self.stride != 1 or self.in_channels != self.out_channels:
            xI = self.convI(xI)

        # add xI to value
        x = x + xI
        x = self.activation(x)

        return x
