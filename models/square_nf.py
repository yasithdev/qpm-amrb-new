import logging
import os
from typing import List

import torch.nn.functional
import torch.optim
import torch.utils.data
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import nf
from .util import gen_img_from_patches, gen_patches_from_img, set_requires_grad


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
):
    model = nf.SquareNormalizingFlow(
        transforms=[
            nf.transforms.AffineCoupling(
                nf.transforms.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.transforms.Conv1x1(**config.conv1x1_config),
            nf.transforms.AffineCoupling(
                nf.transforms.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.transforms.Conv1x1(**config.conv1x1_config),
            nf.transforms.AffineCoupling(
                nf.transforms.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.transforms.Conv1x1(**config.conv1x1_config),
            nf.transforms.AffineCoupling(
                nf.transforms.CouplingNetwork(**config.coupling_network_config)
            ),
            nf.transforms.Conv1x1(**config.conv1x1_config),
        ]
    )

    # set up optimizer
    optim_config = {"params": model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.Adam(**optim_config)

    # load saved model and optimizer, if present
    if config.exc_resume:
        model_state_path = os.path.join(experiment_path, "model.pth")
        optim_state_path = os.path.join(experiment_path, "optim.pth")

        if os.path.exists(model_state_path):
            model.load_state_dict(
                torch.load(model_state_path, map_location=config.device)
            )
            logging.info("Loaded saved model state from:", model_state_path)

        if os.path.exists(optim_state_path):
            optim.load_state_dict(
                torch.load(optim_state_path, map_location=config.device)
            )
            logging.info("Loaded saved optim state from:", optim_state_path)

    return model, optim


def train_model(
    nn: nf.transforms.FlowTransform,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
) -> None:
    # initialize loop
    nn = nn.to(config.device)
    nn.train()
    size = len(config.train_loader.dataset)
    sum_loss = 0

    # training
    set_requires_grad(nn, True)
    with torch.enable_grad():
        iterable = tqdm(
            config.train_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args
        )
        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:
            # reshape
            x = gen_patches_from_img(x.to(config.device), patch_hw=config.patch_hw)
            x = x.float()

            # forward pass
            z, fwd_log_det = nn.forward(x)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            x_r, inv_log_det = nn.inverse(z_pad)

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
    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")
    stats.append(avg_loss)

    # save model/optimizer states
    if not config.exc_dry_run:
        model_state_path = os.path.join(experiment_path, "model.pth")
        optim_state_path = os.path.join(experiment_path, "optim.pth")
        torch.save(nn.state_dict(), model_state_path)
        torch.save(optim.state_dict(), optim_state_path)


def test_model(
    nn: nf.transforms.FlowTransform,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
) -> None:
    # initialize loop
    nn = nn.to(config.device)
    nn.eval()
    size = len(config.test_loader.dataset)
    sum_loss = 0

    # initialize plot(s)
    img_count = 5
    fig, ax = plt.subplots(img_count, 2)
    current_plot = 0

    # testing
    set_requires_grad(nn, False)
    with torch.no_grad():
        iterable = tqdm(
            config.test_loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args
        )
        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:
            # reshape
            x = gen_patches_from_img(x.to(config.device), patch_hw=config.patch_hw)
            x = x.float()

            # forward pass
            z, fwd_log_det = nn.forward(x)
            zu = nf.util.proj(z, manifold_dims=config.manifold_c)
            z_pad = nf.util.pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            x_r, inv_log_det = nn.inverse(z_pad)

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
                    gen_img_from_patches(x, patch_hw=config.patch_hw).to("cpu")[0, 0]
                )
                ax[current_plot, 1].imshow(
                    gen_img_from_patches(x_r, patch_hw=config.patch_hw).to("cpu")[0, 0]
                )
                current_plot += 1

    # post-testing
    avg_loss = sum_loss / size
    stats.append(avg_loss)
    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, f"test_epoch_{epoch}.png"))
    plt.close()
