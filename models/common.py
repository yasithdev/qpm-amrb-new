import logging
import os
from typing import List, Callable, Tuple

import numpy as np
import torch
from config import Config
from einops import rearrange
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, top_k_accuracy_score as topk_acc

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
        # torch.nn.Softmax(dim=1),
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
    model_state_path = os.path.join(experiment_path, f"model.pth")
    optim_state_path = os.path.join(experiment_path, f"optim.pth")

    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path, map_location=config.device))
        logging.info("Loaded saved model state from:", model_state_path)

    if os.path.exists(optim_state_path):
        optim.load_state_dict(torch.load(optim_state_path, map_location=config.device))
        logging.info("Loaded saved optim state from:", optim_state_path)


# --------------------------------------------------------------------------------------------------------------------------------------------------


def save_state(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    experiment_path: str,
    epoch: int,
) -> None:
    logging.info("checkpoint - saving current model and optimizer state")
    model_state_path = os.path.join(experiment_path, f"model_e{epoch}.pth")
    optim_state_path = os.path.join(experiment_path, f"optim_e{epoch}.pth")
    torch.save(model.state_dict(), model_state_path)
    torch.save(optim.state_dict(), optim_state_path)


# --------------------------------------------------------------------------------------------------------------------------------------------------


class Functional(torch.nn.Module):
    """
    Simple module that wraps a parameterless function.

    """

    def __init__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.func = func

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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


def get_conv_out_shape(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    blocks: int = 1,
) -> int:
    size_o = input_size
    for _ in range(blocks):
        size_o = (size_o + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return size_o


# --------------------------------------------------------------------------------------------------------------------------------------------------


def get_convt_out_shape(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    blocks: int = 1,
) -> int:
    size_o = input_size
    for _ in range(blocks):
        size_o = (
            (size_o - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )
    return size_o


# --------------------------------------------------------------------------------------------------------------------------------------------------


def gen_epoch_acc(
    y_pred: list,
    y_true: list,
) -> Tuple[float, float, float]:

    labels = np.arange(len(y_pred[0]))
    top1_acc = float(topk_acc(y_true=y_true, y_score=y_pred, labels=labels, k=1))
    top2_acc = float(topk_acc(y_true=y_true, y_score=y_pred, labels=labels, k=2))
    top3_acc = float(topk_acc(y_true=y_true, y_score=y_pred, labels=labels, k=3))
    return top1_acc, top2_acc, top3_acc


# --------------------------------------------------------------------------------------------------------------------------------------------------


def plot_confusion_matrix(
    cf_matrix: np.ndarray,
    labels: list,
    experiment_path: str,
    epoch: int,
) -> None:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cf_matrix,
        display_labels=labels,
    )
    n = len(labels)
    fig, ax = plt.subplots(figsize=(n + 1, n + 1))
    disp.plot(ax=ax)
    np.save(os.path.join(experiment_path, f"cm_e{epoch}.npy"), cf_matrix)
    plt.savefig(os.path.join(experiment_path, f"cm_e{epoch}.png"))
    plt.close()


# --------------------------------------------------------------------------------------------------------------------------------------------------


def margin_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    m_hi: float = 0.9,
    m_lo: float = 0.1,
    t: float = 0.5,
) -> torch.Tensor:
    loss_hi = torch.clamp(m_hi - y_pred, min=0) ** 2
    loss_lo = torch.clamp(y_pred - m_lo, min=0) ** 2
    loss = y_true * loss_hi + (1 - y_true) * t * loss_lo
    return loss.sum(dim=1).mean()


# --------------------------------------------------------------------------------------------------------------------------------------------------


def gather_samples(
    predictions: List,
    x: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    y_x: torch.Tensor,
    num_predictions: int = 5,
) -> None:

    pattern = "c h w -> h w c"

    if len(predictions) < num_predictions:

        val_x = rearrange(x[0].cpu(), pattern).numpy()
        val_y = y[0].argmax().cpu().numpy()
        val_x_z = rearrange(m[0].detach().cpu(), pattern).numpy()
        val_y_x = y_x[0].detach().argmax().cpu().numpy()

        predictions.append((val_x, val_y, val_x_z, val_y_x))
