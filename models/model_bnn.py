from typing import Tuple

import torch
import torch.utils.data
from config import Config


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer,...]]:

    raise NotImplementedError()


def train_model(
    model: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    optim: Tuple[torch.optim.Optimizer,...],
    **kwargs,
) -> dict:

    raise NotImplementedError()


def test_model(
    model: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    **kwargs,
) -> dict:

    raise NotImplementedError()
