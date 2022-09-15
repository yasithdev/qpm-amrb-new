from typing import Callable, Tuple

import torch
from config import Config


def get_model_optimizer_and_loops(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Callable, Callable]:
    # load requested module, if available
    if config.model_name == "nf":
        from .model_nf import load_model_and_optimizer, test_model, train_model
    elif config.model_name == "resnet":
        from .model_resnet import load_model_and_optimizer, test_model, train_model
    elif config.model_name == "capsnet":
        from .model_capsnet import load_model_and_optimizer, test_model, train_model
    elif config.model_name == "efficientcaps":
        from .model_efficientcaps import (
            load_model_and_optimizer,
            test_model,
            train_model,
        )
    elif config.model_name == "deepcaps":
        from .model_deepcaps import load_model_and_optimizer, test_model, train_model
    elif config.model_name == "bnn":
        from .model_bnn import load_model_and_optimizer, test_model, train_model
    else:
        raise ImportError(f"Model not found: {config.model_name}")
    # return model, optimizer, training loop, and testing loop
    model, optimizer = load_model_and_optimizer(config, experiment_path)
    return model, optimizer, train_model, test_model
