import os
from typing import List

import torch.nn.functional
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

import dflows.transforms as nft
from config import Config
from dflows.flow_util import pad, proj
from dflows.img_util import gen_img_from_patches, gen_patches_from_img
from dflows.nn_util import set_requires_grad


def train_model(
    nn: nft.FlowTransform,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    optim: torch.optim.Optimizer,
    train_losses: List,
    experiment_path: str,
) -> None:
    # initialize loop
    nn = nn.to(config.device)
    nn.train()
    size = len(loader.dataset)
    sum_loss = 0

    # training
    set_requires_grad(nn, True)
    with torch.enable_grad():
        iterable = tqdm(loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:
            # reshape
            x = gen_patches_from_img(x.to(config.device), patch_hw=config.patch_hw)
            x = x.float()

            # forward pass
            z, fwd_log_det = nn.forward(x)
            zu = proj(z, manifold_dims=config.manifold_c)
            z_pad = pad(
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
            stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(stats)

    # post-training
    avg_loss = sum_loss / size
    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")
    train_losses.append(avg_loss)

    # save model/optimizer states
    if not config.exc_dry_run:
        model_state_path = os.path.join(experiment_path, "model.pth")
        optim_state_path = os.path.join(experiment_path, "optim.pth")
        torch.save(nn.state_dict(), model_state_path)
        torch.save(optim.state_dict(), optim_state_path)


def test_model(
    nn: nft.FlowTransform,
    epoch: int,
    loader: torch.utils.data.DataLoader,
    config: Config,
    test_losses: List,
    experiment_path: str,
) -> None:
    # initialize loop
    nn = nn.to(config.device)
    nn.eval()
    size = len(loader.dataset)
    sum_loss = 0

    # initialize plot(s)
    img_count = 5
    fig, ax = plt.subplots(img_count, 2)
    current_plot = 0

    # testing
    set_requires_grad(nn, False)
    with torch.no_grad():
        iterable = tqdm(loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args)
        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:
            # reshape
            x = gen_patches_from_img(x.to(config.device), patch_hw=config.patch_hw)
            x = x.float()

            # forward pass
            z, fwd_log_det = nn.forward(x)
            zu = proj(z, manifold_dims=config.manifold_c)
            z_pad = pad(
                zu,
                off_manifold_dims=(config.input_chw[0] - config.manifold_c),
            )
            x_r, inv_log_det = nn.inverse(z_pad)

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(stats)

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
    test_losses.append(avg_loss)
    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, "test_epoch_{epoch}.png"))
    plt.close()
