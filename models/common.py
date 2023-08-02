import logging
import os
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.hooks
from einops import rearrange
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import top_k_accuracy_score as topk_acc

from config import Config

# --------------------------------------------------------------------------------------------------------------------------------------------------


def load_model_state(
    model: torch.nn.Module,
    config: Config,
    epoch: Optional[int] = None,
) -> None:
    if epoch is not None:
        model_filename = f"{config.model_name}_model_e{epoch}.pth"
    else:
        model_filename = f"{config.model_name}_model.pth"

    model_state_path = os.path.join(config.experiment_path, model_filename)

    if not os.path.exists(model_state_path):
        logging.warn("no saved model state: %s", model_state_path)
    else:
        logging.info("loading saved model state from: %s", model_state_path)
        model.load_state_dict(torch.load(model_state_path, map_location=config.device))


# --------------------------------------------------------------------------------------------------------------------------------------------------


def save_model_state(
    model: torch.nn.Module,
    config: Config,
    epoch: int,
) -> None:
    logging.info(f"saving model state - e{epoch}")
    model_filename = f"{config.model_name}_model_e{epoch}.pth"
    model_state_path = os.path.join(config.experiment_path, model_filename)
    torch.save(model.state_dict(), model_state_path)


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
    n: int = 1,
) -> int:
    size_o = input_size
    for _ in range(n):
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


def gen_topk_accs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_labels: int,
) -> dict[str, float]:
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    pseudo_labels = np.arange(num_labels)
    top1_acc = topk_acc(y_true, y_pred, labels=pseudo_labels, k=1)
    top2_acc = topk_acc(y_true, y_pred, labels=pseudo_labels, k=2)
    top3_acc = topk_acc(y_true, y_pred, labels=pseudo_labels, k=3)
    return dict(acc1=top1_acc, acc2=top2_acc, acc3=top3_acc)


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
    return loss.sum(dim=-1).mean()


# --------------------------------------------------------------------------------------------------------------------------------------------------


def gather_samples(
    samples: List[Tuple[np.ndarray, int, np.ndarray, int]],
    x: torch.Tensor,
    y: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    num_samples: int = 5,
) -> None:
    pattern = "c h w -> h w c"

    if len(samples) < num_samples:
        item_x = rearrange(x[0].cpu(), pattern).numpy()
        item_y = int(y[0].cpu().numpy())
        item_xp = rearrange(xp[0].detach().cpu(), pattern).numpy()
        item_yp = int(yp[0].detach().argmax().cpu().numpy())

        samples.append((item_x, item_y, item_xp, item_yp))


# --------------------------------------------------------------------------------------------------------------------------------------------------


class GradientScaler(torch.autograd.Function):
    factor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        factor = GradientScaler.factor
        return factor.view(-1, 1, 1, 1) * grad_output


# --------------------------------------------------------------------------------------------------------------------------------------------------


def get_gradient_ratios(
    lossA: torch.Tensor,
    lossB: torch.Tensor,
    x_f: torch.Tensor,
) -> torch.Tensor:
    grad = torch.autograd.grad
    grad_lossA_xf = grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma


# --------------------------------------------------------------------------------------------------------------------------------------------------
class GradientHook(object):
    def __init__(
        self,
        module: torch.nn.Module,
        is_negate: bool = True,
    ) -> None:
        self.module = module
        self.is_negate = is_negate
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def set_negate(self, is_negate):
        self.is_negate = is_negate

    def negate_grads_func(self, module, grad_input, grad_output):
        # if is_negate is false, not negate grads
        if not self.is_negate:
            return

        if isinstance(grad_input, tuple):
            result = []
            for item in grad_input:
                if isinstance(item, torch.Tensor):
                    result.append(torch.neg(item))
                else:
                    result.append(item)
            return tuple(result)
        if isinstance(grad_input, torch.Tensor):
            return torch.neg(grad_input)

    def set_negate_grads_hook(self):
        self.handle = self.module.register_full_backward_hook(self.negate_grads_func)

    def __del__(self):
        if self.handle:
            self.handle.remove()


def edl_kl(
    α: torch.Tensor,
) -> torch.Tensor:
    # function definitions
    lnΓ = torch.lgamma
    ψ = torch.digamma  # derivative of lnΓ
    Σ = partial(torch.sum, dim=-1, keepdim=True)

    # logic
    β = α.new_ones(α.size())
    l1 = lnΓ(Σ(α)) - Σ(lnΓ(α)) + Σ(lnΓ(β)) - lnΓ(Σ(β))
    l2 = Σ((α - β) * (ψ(α) - ψ(Σ(α))))

    return l1 + l2


def edl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    q: int = 10,
    τ: int = 10,
    λ_max: float = 1.0,
) -> torch.Tensor:
    # function definitions
    Σ = partial(torch.sum, dim=-1, keepdim=True)
    h = lambda x: torch.as_tensor(F.relu(x))
    (B, K) = logits.size()

    # logic
    e = h(logits)
    α = e + 1
    S = Σ(α)
    p = α / S

    # make y one-hot
    if target.dim() == 2:
        assert (B, K) == target.size()
        y = target.float()
    else:
        assert B == target.size(0)
        y = logits.new_zeros(B, K)
        idx_ind = (target < K).nonzero()
        if len(idx_ind) > 0:
            y[idx_ind] = F.one_hot(target[idx_ind], K).float()

    # wait for q epochs, then anneal λ from 0 -> λ_max over τ epochs
    λ = min(λ_max, max(epoch - q, 0) / τ)
    α̃ = (1 - y) * (α - 1) + 1

    L_err = Σ((y - p).pow(2))
    L_var = Σ(p * (1 - p) / (S + 1))
    L_kl = λ * edl_kl(α̃)

    return L_err + L_var + L_kl


def edl_probs(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # function definitions
    Σ = partial(torch.sum, dim=-1, keepdim=True)
    h = lambda x: torch.as_tensor(F.relu(x))
    K = logits.size(-1)

    # logic
    e = h(logits)
    α = e + 1
    S = Σ(α)
    p = α / S

    b = e / S
    u = K / S

    return p, u
