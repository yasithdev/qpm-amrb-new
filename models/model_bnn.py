from typing import Optional, Tuple

import torch
import torch.utils.data

from config import Config


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:
    raise NotImplementedError()


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:
    raise NotImplementedError()


def step_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
    **kwargs,
) -> dict:
    raise NotImplementedError()
