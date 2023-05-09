import torch

from typing import Union, Type, Tuple

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


def get_encoder(
    input_chw: Tuple[int, int, int],
    num_features: int,
) -> torch.nn.Module:
    """
    Create a ResNet-Based Encoder

    :param input_chw: (C, H, W)
    :param num_features: number of output channels (D)
    :return: model that transforms (B, C, H, W) -> (B, D, 1, 1)
    """
    (c, h, w) = input_chw
    d = num_features

    k1, s1 = 5, 2
    pw1, ph1 = npad(w, k1, s1), npad(h, k1, s1)
    w1, h1 = (w - k1 + pw1 + 1) // s1, (h - k1 + pw1 + 1) // s1

    c1 = 32
    c2 = 48
    c3 = 64

    model = torch.nn.Sequential(
        # (B,C,H,W) -> (B,C,H+ph,W+pw)
        torch.nn.ZeroPad2d((0, pw1, 0, ph1)),
        # (B,C,H+ph,W+pw) -> (B,C1,H/2,W/2)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=c1,
            kernel_size=k1,
            stride=s1,
            bias=False,
        ),
        torch.nn.GroupNorm(4, c1),
        torch.nn.LeakyReLU(),
        # (B,C1,H/2,W/2) -> (B,C1,H/2,W/2)
        ResidualBlock(
            in_channels=c1,
            hidden_channels=c1,
            out_channels=c1,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.GroupNorm,
        ),
        # (B,C1,H/2,W/2) -> (B,C2,H/4,W/4)
        ResidualBlock(
            in_channels=c1,
            hidden_channels=c1,
            out_channels=c2,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.GroupNorm,
        ),
        # (B,C2,H/4,W/4) -> (B,C2,H/4,W/4)
        ResidualBlock(
            in_channels=c2,
            hidden_channels=c2,
            out_channels=c2,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.GroupNorm,
        ),
        # (B,C2,H/4,W/4) -> (B,C3,H/8,W/8)
        ResidualBlock(
            in_channels=c2,
            hidden_channels=c2,
            out_channels=c3,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.GroupNorm,
        ),
        # (B,C3,H/8,W/8) -> (B,D,1,1)
        torch.nn.Conv2d(
            in_channels=c3,
            out_channels=d,
            kernel_size=(h // 8, w // 8),
            stride=1,
        ),
        torch.nn.Flatten(),
    )

    return model
