
from typing import Tuple, Callable
import torch

from config import Config

def get_model_optimizer_and_loops(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Callable, Callable]:
    # load requested module, if available
    if config.model_name == "nf":
        from .model_nf import load_model_and_optimizer, train_model, test_model
    elif config.model_name == "resnet":
        from .model_resnet import load_model_and_optimizer, train_model, test_model
    elif config.model_name == "capsnet":
        from .model_capsnet import load_model_and_optimizer, train_model, test_model
    else:
        raise ImportError(f"Model not found: {config.model_name}")
    # return model, optimizer, training loop, and testing loop
    model, optimizer = load_model_and_optimizer(config, experiment_path)
    return model, optimizer, train_model, test_model