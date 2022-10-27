from typing import Callable, Tuple

import torch
import torchinfo
from config import Config


def get_model_optimizer_and_loops(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Callable, Callable]:

    # load requested module, if available
    input_size = (config.batch_size, *config.image_chw)
    if config.model_name == "nf":
        from .model_nf import load_model_and_optimizer, test_model, train_model

        input_size = (config.batch_size, *config.input_chw)
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
    elif config.model_name == "drcaps":
        from .model_drcaps import load_model_and_optimizer, test_model, train_model
    elif config.model_name == "bnn":
        from .model_bnn import load_model_and_optimizer, test_model, train_model
    else:
        raise ImportError(f"Model not found: {config.model_name}")

    # load model and optimizer to device
    model, optimizer = load_model_and_optimizer(config, experiment_path)
    model = model.to(config.device)

    # print model summary

    # normalizing flow model
    if config.model_name == "nf":
        torchinfo.summary(model, input_size=input_size, depth=5)

    # resnet model
    elif config.model_name == "resnet":
        C = 128
        enc_out_size = (config.batch_size, C, 1, 1)
        cls_in_size = enc_out_size
        dec_in_size = enc_out_size
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # capsnet model
    elif config.model_name == "capsnet":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (config.batch_size, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (config.batch_size, C, 1, 1)
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # efficientcaps model
    elif config.model_name == "efficientcaps":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (config.batch_size, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (config.batch_size, C, 1, 1)
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # dynamic routing caps model
    elif config.model_name == "drcaps":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (config.batch_size, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (config.batch_size, C, 1, 1)
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # bnn model
    elif config.model_name == "bnn":
        raise NotImplementedError("Model not implemented")

    else:
        raise ImportError(f"Model not found: {config.model_name}")

    # return model, optimizer, training loop, and testing loop
    return model, optimizer, train_model, test_model
