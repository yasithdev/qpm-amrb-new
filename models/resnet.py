from typing import List, Tuple

import torch
import torch.utils.data
from config import Config

from .resnet import ResidualBlock


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
):
    # raise NotImplementedError()

    num_features=128

    # BCHW -> BD11
    encoder = get_encoder(
        input_chw=config.input_chw,
        num_features=num_features,
    )

    # BD11 -> BCHW
    decoder = get_decoder(
        num_features=num_features,
        output_chw=config.input_chw,
    )

    # BD11 -> BL
    classifier = get_classifier(
        in_features=num_features,
        out_features=config.
    )


def train_model(
    nn: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
) -> None:
    raise NotImplementedError()


def test_model(
    nn: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    stats: List,
    experiment_path: str,
) -> None:
    raise NotImplementedError()


# ------------------------ COPIED FROM TF IMPLEMENTATION -----------------------------------


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

    model = torch.nn.Sequential(
        # (B, C, H, W)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=d // 4,
            kernel_size=7,
            stride=2,
            padding="same",
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
            kernel_size=(h // 8, w // 8),
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
            stride=2,
            padding="same",
        ),
        # (B, C, H, W)
    )
    return model


def get_classifier(
    in_features: int,
    out_features: int,
) -> torch.nn.Module:
    """
    Simple classifier with a dense layer + softmax activation

    :param in_features: number of input channels (D)
    :param out_features: number of output features (L)
    :return: model that transforms (B, D, 1, 1) -> (B, L)
    """
    d = in_features

    model = torch.nn.Sequential(
        # (B, D, 1, 1)
        torch.nn.Flatten(),
        # (B, D)
        torch.nn.Linear(in_features=d, out_features=out_features),
        # (B, L)
        torch.nn.Softmax(dim=1),
        # (B, L)
    )
    return model
