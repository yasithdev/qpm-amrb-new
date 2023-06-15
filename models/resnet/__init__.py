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
    :return: model that transforms (B, C, H, W) -> (B, D, 1, 1)
    """
    (c, h, w) = input_chw
    d = num_features
    assert h % 8 == 0
    assert w % 8 == 0

    model = torch.nn.Sequential(
        # (B, c, h, w)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=d // 4,
            kernel_size=7,
            padding=3,
            stride=2,
        ),
        # (B, d/4, h/2, w/2)
        torch.nn.GroupNorm(1, d // 4),
        # (B, d/4, h/2, w/2)
        torch.nn.ReLU(),
        # (B, d/4, h/2, w/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 4,
            stride=1,
            conv=torch.nn.Conv2d,
        ),
        # (B, d/4, h/2, w/2)
        ResidualBlock(
            in_channels=d // 4,
            hidden_channels=d // 4,
            out_channels=d // 2,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B, d/2, h/4, w/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d // 2,
            stride=1,
            conv=torch.nn.Conv2d,
        ),
        # (B, d/2, h/4, w/4)
        ResidualBlock(
            in_channels=d // 2,
            hidden_channels=d // 2,
            out_channels=d,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B, d, h/8, w/8)
        torch.nn.Conv2d(
            in_channels=d,
            out_channels=d,
            groups=d,
            kernel_size=(h // 8, w // 8),
        ),
        # (B, d, 1, 1)
        torch.nn.GroupNorm(1, d),
        # (B, d, 1, 1)
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
    assert h % 8 == 0
    assert w % 8 == 0

    model = torch.nn.Sequential(
        # (B, D, 1, 1)
        torch.nn.ConvTranspose2d(
            in_channels=d,
            out_channels=d,
            groups=d,
            kernel_size=(h // 8, w // 8),
        ),
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
            out_channels=d // 4,
            stride=1,
            conv=torch.nn.ConvTranspose2d,
        ),
        # (B, D/4, H/2, W/2)
        torch.nn.ConvTranspose2d(
            in_channels=d // 4,
            out_channels=c,
            kernel_size=7,
            padding=3,
            stride=2,
            output_padding=1,
        ),
        # (B, C, H, W)
        # crop for consistency
    )
    return model
