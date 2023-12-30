from typing import Tuple

import torch

from .residual_block import ResidualBlock


def get_encoder(
    input_chw: Tuple[int, int, int],
    num_features: int,
) -> torch.nn.Module:
    """
    Create a ResNet-Based Encoder

    :param input_chw: (C, H, W)
    :param num_features: number of output channels (D)
    :return: model that transforms (B, C, H, W) -> (B, D, H/8, W/8)
    """
    (c, h, w) = input_chw
    d = num_features
    assert h % 8 == 0
    assert w % 8 == 0

    model = torch.nn.Sequential(
        # (B, C, H, W)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=2,
        ),
        # (B, 64, H/2, W/2)
        torch.nn.ReLU(),
        ResidualBlock(
            in_channels=64,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=1,
            conv=torch.nn.Conv2d,
        ),
        # (B, D/4, H/2, W/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 2,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=1,
            conv=torch.nn.Conv2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B, D, H/8, W/8)
    )

    return model


def get_decoder(
    num_features: int,
    output_chw: Tuple[int, int, int],
) -> torch.nn.Module:
    """

    :param num_features: Number of input channels (D)
    :param output_chw:  (C, H, W)
    :return: model that transforms (B, D, H/8, W/8) -> (B, C, H, W)
    """
    d = num_features
    (c, h, w) = output_chw
    assert h % 8 == 0
    assert w % 8 == 0

    model = torch.nn.Sequential(
        # (B, D, H/8, W/8)
        ResidualBlock(
            in_channels=d,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=2,
            conv=torch.nn.ConvTranspose2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=1,
            conv=torch.nn.ConvTranspose2d,
        ),
        # (B, D/2, H/4, W/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=2,
            conv=torch.nn.ConvTranspose2d,
        ),
        # (B, D/4, H/2, W/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=64,
            stride=1,
            conv=torch.nn.ConvTranspose2d,
        ),
        # (B, 64, H/2, W/2)
        torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=c,
            kernel_size=3,
            padding=1,
            stride=2,
            output_padding=1,
        ),
        # (B, C, H, W)
    )
    return model
