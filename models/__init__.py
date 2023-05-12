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

    if config.model_name == "flow_gan":
        from .model_flow_gan import load_model_and_optimizer, test_model, train_model

    elif config.model_name == "flow_wgan":
        from .model_flow_wgan import load_model_and_optimizer, test_model, train_model
    
    elif config.model_name == "flow_cgan":
        from .model_flow_cgan import load_model_and_optimizer, test_model, train_model

    elif config.model_name == "flow_cwgan":
        from .model_flow_cwgan import load_model_and_optimizer, test_model, train_model
    
    elif config.model_name == "flow_nll":
        from .model_flow_nll import load_model_and_optimizer, test_model, train_model

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
    if config.model_name[:4] == "flow":
        x_size = (B, C, H, W)
        c, h, w = C * 64, H // 8, W // 8
        cm = config.manifold_d // h // w
        u_size = (B, cm * h * w)
        # x_inv_size = (B, c, h, w)
        torchinfo.summary(model["flow"], input_size=x_size, depth=5)
        # torchinfo.summary(model["x_flow"], input_size=x_size, depth=5)

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
