from typing import Callable, Tuple

import torch
import torchinfo
from config import Config


def get_model_optimizer_and_loops(
    config: Config,
) -> Tuple[torch.nn.Module, Tuple[torch.optim.Optimizer,...], Callable, Callable]:

    # load requested module, if available
    assert config.image_chw
    assert config.dataset_info
    B, C, H, W = (config.batch_size, *config.image_chw)

    if config.model_name == "flow":
        from .model_flow import load_model_and_optimizer, test_model, train_model

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
        raise NotImplementedError("Model not implemented")

    else:
        raise ImportError(f"Model not found: {config.model_name}")

    # load model and optimizer to device
    model, optimizer = load_model_and_optimizer(config)
    model = model.float().to(config.device)

    # print model summary
    input_size = (B, C, H, W)

    # flow model
    if config.model_name == "flow":
        x_size = (B, C, H, W)
        h, w = H // 8, W // 8
        c = config.manifold_d // h // w
        u_size = B, c, h, w
        torchinfo.summary(model["x_flow"], input_size=x_size, depth=5)
        torchinfo.summary(model["u_flow"], input_size=u_size, depth=5)

    # resnet model
    elif config.model_name == "resnet":
        C = 128
        enc_out_size = (B, C, 1, 1)
        cls_in_size = enc_out_size
        dec_in_size = enc_out_size
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # capsnet model
    elif config.model_name == "capsnet":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (B, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (B, C, 1, 1)
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # efficientcaps model
    elif config.model_name == "efficientcaps":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (B, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (B, C, 1, 1)
        torchinfo.summary(model["encoder"], input_size=input_size, depth=5)
        torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
        torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)

    # dynamic routing caps model
    elif config.model_name == "drcaps":
        C, N = 16, config.dataset_info["num_train_labels"]
        enc_out_size = (B, C, N)
        cls_in_size = enc_out_size
        dec_in_size = (B, C, 1, 1)
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
