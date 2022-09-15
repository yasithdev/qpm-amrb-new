import logging
import os
from typing import Tuple

import torch
from config import Config
from einops import rearrange

# --------------------------------------------------------------------------------------------------------------------------------------------------


def get_classifier(
    in_features: int,
    out_features: int,
) -> torch.nn.Module:
    """
    Simple classifier with a dense layer + softmax activation

    :param in_features: number of input channels (D)
    :param out_features: number of output features (L)
    :return: model that transforms (B, D, 1, 1) -> (B, L)
    """
    d = in_features

    model = torch.nn.Sequential(
        # (B, D, 1, 1)
        torch.nn.Flatten(),
        # (B, D)
        torch.nn.Linear(in_features=d, out_features=out_features),
        # (B, L)
        torch.nn.Softmax(dim=1),
        # (B, L)
    )
    return model


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load_saved_state(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    experiment_path: str,
    config: Config,
) -> None:
    model_state_path = os.path.join(experiment_path, "model.pth")
    optim_state_path = os.path.join(experiment_path, "optim.pth")

    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path, map_location=config.device))
        logging.info("Loaded saved model state from:", model_state_path)

    if os.path.exists(optim_state_path):
        optim.load_state_dict(torch.load(optim_state_path, map_location=config.device))
        logging.info("Loaded saved optim state from:", optim_state_path)


# --------------------------------------------------------------------------------------------------------------------------------------------------


class Functional(torch.nn.Module):
    """
    Simple module that wraps a parameterless function.

    """

    def __init__(self, func):
        super().__init__()
        from types import LambdaType

        assert type(func) is LambdaType
        self.func = func

    def forward(self, x):
        return self.func(x)


# --------------------------------------------------------------------------------------------------------------------------------------------------


def gen_patches_from_img(
    x: torch.Tensor,
    patch_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Generate patches for a set of images

    :param x: Batch of Images
    :param patch_hw: Patch Height and Width
    :return:
    """
    patched = rearrange(
        x, "b c (h ph) (w pw) -> b (ph pw c) h w", ph=patch_hw[0], pw=patch_hw[1]
    )
    return patched


# --------------------------------------------------------------------------------------------------------------------------------------------------


def gen_img_from_patches(
    x: torch.Tensor,
    patch_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Generate back images from a set of image patches

    :param x: Batch of Patched Images
    :param patch_h: Patch Height
    :param patch_w: Patch Width
    :return:
    """
    unpatched = rearrange(
        x, "b (ph pw c) h w -> b c (h ph) (w pw)", ph=patch_hw[0], pw=patch_hw[1]
    )
    return unpatched


# --------------------------------------------------------------------------------------------------------------------------------------------------


def set_requires_grad(module: torch.nn.Module, flag: bool):
    """
    Sets requires_grad flag of all parameters of a torch.nn.module
    :param module: torch.nn.module
    :param flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag


# --------------------------------------------------------------------------------------------------------------------------------------------------
