import importlib
from typing import Callable, Tuple

import torch

from config import Config


def get_model_optimizer_and_loops(
    config: Config,
) -> Tuple[torch.nn.Module, Tuple[torch.optim.Optimizer, ...], Callable, Callable]:

    # load requested module, if available
    assert config.image_chw
    assert config.dataset_info

    try:
        module = importlib.__import__(
            name=f"model_{config.model_name}",
            globals=globals(),
            fromlist=[
                "load_model_and_optimizer",
                "describe_model",
                "train_model",
                "test_model",
            ],
            level=1,
        )
    except ImportError as e:
        raise e

    # load model and optimizer to device
    model, optimizer = module.load_model_and_optimizer(config)
    model = model.float().to(config.device)

    # print model summary
    module.describe_model(model)

    # return model, optimizer, training loop, and testing loop
    return model, optimizer, module.train_model, module.test_model
