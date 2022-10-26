from typing import Tuple

import torch

from ..common import Functional
from .residual_block import ResidualBlock


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

    rh = 8 - (h % 8)
    rw = 8 - (w % 8)
    ph1, pw1 = rh // 2, rw // 2
    ph2, pw2 = rh - ph1, rw - pw1

    model = torch.nn.Sequential(
        # zero-pad for consistency
        torch.nn.ZeroPad2d((pw1, pw2, ph1, ph2)),
        # (B, C, H, W)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=d // 4,
            kernel_size=7,
            padding=3,
            stride=2,
            bias=False,
        ),
        # (B, D/4, H/2, W/2)
        torch.nn.BatchNorm2d(
            num_features=d // 4,
        ),
        # (B, D/4, H/2, W/2)
        torch.nn.ReLU(),
        # (B, D/4, H/2, W/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/4, H/2, W/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 2,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D, H/8, W/8)
        torch.nn.Conv2d(
            in_channels=d,
            out_channels=d,
            groups=d,
            kernel_size=((h + rh) // 8, (w + rw) // 8),
            bias=False,
        ),
        # (B, D, 1, 1)
        torch.nn.BatchNorm2d(
            num_features=d,
        ),
        # (B, D, 1, 1)
    )

    return model


def get_decoder(
    num_features: int,
    output_chw: Tuple[int, int, int],
) -> torch.nn.Module:
    """

    :param num_features: Number of input channels (D)
    :param output_chw:  (C, H, W)
    :return: model that transforms (B, D, 1, 1) -> (B, C, H, W)
    """
    d = num_features
    (c, h, w) = output_chw

    rh = 8 - (h % 8)
    rw = 8 - (w % 8)
    ph1, pw1 = rh // 2, rw // 2
    ph2, pw2 = rh - ph1, rw - pw1

    model = torch.nn.Sequential(
        # (B, D, 1, 1)
        torch.nn.ConvTranspose2d(
            in_channels=d,
            out_channels=d,
            groups=d,
            kernel_size=((h + rh) // 8 + 1, (w + rw) // 8 + 1),
        ),
        # (B, D, H/8, W/8)
        ResidualBlock(
            in_channels=d,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=2,
            conv=torch.nn.ConvTranspose2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=1,
            conv=torch.nn.ConvTranspose2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=2,
            conv=torch.nn.ConvTranspose2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/4, H/2, W/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=1,
            conv=torch.nn.ConvTranspose2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B, D/4, H/2, W/2)
        torch.nn.ConvTranspose2d(
            in_channels=d // 4,
            out_channels=c,
            kernel_size=7,
            padding=3,
            stride=2,
        ),
        # (B, C, H, W)
        Functional(lambda x: x[..., ph1 : -ph2 - 1, pw1 : -pw2 - 1]),
        # crop for consistency
    )
    return model
