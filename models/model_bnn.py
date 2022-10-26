from typing import List, Tuple

import torch
import torch.utils.data
from config import Config


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:

    raise NotImplementedError()


def train_model(
    model: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> dict:

    raise NotImplementedError()


def test_model(
    model: torch.nn.Module,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> dict:

    raise NotImplementedError()
