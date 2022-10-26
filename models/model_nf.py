import os
from typing import List, Tuple

import torch
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import nf
from .common import (
    gen_epoch_stats,
    gen_img_from_patches,
    gen_patches_from_img,
    load_saved_state,
    plot_confusion_matrix,
    save_state,
    set_requires_grad,
)


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
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

    # load saved model and optimizer, if present
    if config.exc_resume:
        load_saved_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
            config=config,
        )

    return model, optim


def train_model(
    model: nf.ops.FlowTransform,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> dict:

    # initialize loop
    model = model.to(config.device)
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

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # reshape
            x = gen_patches_from_img(x, patch_hw=config.patch_hw)

            # forward pass
            z, fwd_log_det = model.forward(x)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            x_r, inv_log_det = model.inverse(z_pad)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())

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
    cf_matrix, acc_score = gen_epoch_stats(y_pred=y_pred, y_true=y_true)
    stats.append(avg_loss)

    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    # save model/optimizer states
    if min(stats) == avg_loss and not config.exc_dry_run:
        save_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
        )

    return {
        "train_loss": avg_loss,
        "train_acc": acc_score,
    }


def test_model(
    model: nf.ops.FlowTransform,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> dict:

    # initialize loop
    model = model.to(config.device)
    model.eval()
    size = len(config.test_loader.dataset)
    sum_loss = 0

    # initialize plot(s)
    img_count = 5
    fig, ax = plt.subplots(img_count, 2)
    current_plot = 0

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

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # reshape
            x = gen_patches_from_img(x.to(config.device), patch_hw=config.patch_hw)

            # forward pass
            z, fwd_log_det = model.forward(x)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            x_r, inv_log_det = model.inverse(z_pad)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

            # accumulate plots
            if current_plot < img_count:
                ax[current_plot, 0].imshow(
                    gen_img_from_patches(x, patch_hw=config.patch_hw)
                    .cpu()
                    .numpy()[0, 0]
                )
                ax[current_plot, 1].imshow(
                    gen_img_from_patches(x_r, patch_hw=config.patch_hw)
                    .cpu()
                    .numpy()[0, 0]
                )
                current_plot += 1

    # post-testing
    avg_loss = sum_loss / size
    cf_matrix, acc_score = gen_epoch_stats(y_pred=y_pred, y_true=y_true)
    stats.append(avg_loss)

    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, f"test_e{epoch}.png"))
    plt.close()

    # plot confusion matrix
    plot_confusion_matrix(
        cf_matrix=cf_matrix,
        labels=config.train_loader.dataset.labels,
        experiment_path=experiment_path,
        epoch=epoch,
    )

    return {
        "test_loss": avg_loss,
        "test_acc": acc_score,
    }
