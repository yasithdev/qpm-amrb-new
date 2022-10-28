from typing import Tuple

import torch
from config import Config
from tqdm import tqdm

from . import nf
from .common import (
    gather_samples,
    gen_epoch_acc,
    gen_img_from_patches,
    gen_patches_from_img,
    set_requires_grad,
)


def load_model_and_optimizer(
    config: Config,
) -> Tuple[nf.ops.FlowTransform, torch.optim.Optimizer]:
    model = nf.SquareNormalizingFlow(
        transforms=[
            nf.ops.AffineCoupling(
                nf.ops.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.ops.Conv1x1(**config.conv1x1_config),
            nf.ops.AffineCoupling(
                nf.ops.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.ops.Conv1x1(**config.conv1x1_config),
            nf.ops.AffineCoupling(
                nf.ops.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.ops.Conv1x1(**config.conv1x1_config),
            nf.ops.AffineCoupling(
                nf.ops.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.ops.Conv1x1(**config.conv1x1_config),
        ]
    )

    # set up optimizer
    optim_config = {"params": model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.Adam(**optim_config)

    return model, optim


def train_model(
    model: nf.ops.FlowTransform,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    **kwargs,
) -> dict:

    # initialize loop
    model = model.float().to(config.device)
    model.train()
    size = len(config.train_loader.dataset)
    sum_loss = 0

    # training
    set_requires_grad(model, True)
    with torch.enable_grad():
        iterable = tqdm(
            config.train_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args
        )
        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            xp = gen_patches_from_img(x, patch_hw=config.patch_hw)
            z, fwd_log_det = model.forward(xp)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            xp_r, inv_log_det = model.inverse(z_pad)
            x_r = gen_img_from_patches(xp_r, patch_hw=config.patch_hw)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

            # backward pass
            optim.zero_grad()
            minibatch_loss.backward()
            optim.step()

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-training
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": samples,
    }


def test_model(
    model: nf.ops.FlowTransform,
    epoch: int,
    config: Config,
    **kwargs,
) -> dict:

    # initialize loop
    model = model.float().to(config.device)
    model.eval()
    size = len(config.test_loader.dataset)
    sum_loss = 0

    # testing
    set_requires_grad(model, False)
    with torch.no_grad():
        iterable = tqdm(
            config.test_loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args
        )
        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            xp = gen_patches_from_img(x, patch_hw=config.patch_hw)
            z, fwd_log_det = model.forward(xp)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            xp_r, inv_log_det = model.inverse(z_pad)
            x_r = gen_img_from_patches(xp_r, patch_hw=config.patch_hw)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-testing
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": samples,
    }
