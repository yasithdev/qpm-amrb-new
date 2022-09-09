
from typing import Tuple, Callable
import torch

from config import Config

def get_model_optimizer_and_loops(
    name: str,
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Callable, Callable]:
    # load requested module, if available
    if name == "nf":
        from .model_nf import load_model_and_optimizer, train_model, test_model
    elif name == "resnet":
        from .model_resnet import load_model_and_optimizer, train_model, test_model
    elif name == "capsnet":
        from .model_capsnet import load_model_and_optimizer, train_model, test_model
    else:
        raise ImportError(f"Model not found: {name}")
    # return model, optimizer, training loop, and testing loop
    model, optimizer = load_model_and_optimizer(config, experiment_path)
    return model, optimizer, train_model, test_model