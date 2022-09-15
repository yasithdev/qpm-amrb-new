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
    nn: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
    raise NotImplementedError()


def test_model(
    nn: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
    raise NotImplementedError()
