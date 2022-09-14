from typing import List

import torch
import torch.utils.data
from config import Config

from .capsnet.ops.caps import ConvCaps2D, FlattenCaps, LinearCaps, MaskCaps, squash


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
):
    head = torch.nn.Conv2d(
        in_channels=config.image_chw[0],
        out_channels=256,
        kernel_size=(9, 9),
        padding="same",
    )
    torch.nn.

    raise NotImplementedError()


def train_model(
    nn: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
    raise NotImplementedError()


def test_model(
    nn: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
    raise NotImplementedError()
