import logging
import math
import os
from functools import partial
from typing import List, Tuple

import torch
import torch.utils.data
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import ConvCaps3D, LinearCaps, MaskCaps, squash
from .capsnet.caps import FlattenCaps
from .common import Functional, load_saved_state, set_requires_grad
from .resnet import get_decoder

caps_shape = (8, 32)
kernel_size = (3, 3)

def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:

    num_labels = config.dataset_info["num_train_labels"]
    flat_caps_shape = (
        caps_shape[0],
        caps_shape[1] * config.image_chw[1] * config.image_chw[2],
    )
    out_caps_shape = (16, num_labels)
    num_features = math.prod(out_caps_shape)

    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=config.image_chw[0],
            out_channels=math.prod(caps_shape),
            kernel_size=kernel_size,
            padding="same",
        ),
        Functional(partial(conv_to_caps, out_capsules=caps_shape)),
        Functional(squash),
        ConvCaps3D(
            in_capsules=caps_shape,
            out_capsules=caps_shape,
            kernel_size=kernel_size,
            padding="same",
        ),
        Functional(squash),
        FlattenCaps(),
        LinearCaps(
            in_channels=flat_caps_shape[0],
            out_channels=out_caps_shape[0],
        ),
        Functional(squash),
    )

    classifier = MaskCaps()

    decoder = get_decoder(
        num_features=num_features,
        output_chw=config.image_chw,
    )

    model = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "classifier": classifier,
            "decoder": decoder,
        }
    )

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
    nn: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
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
        encoder = nn["encoder"]
        classifier = nn["classifier"]
        decoder = nn["decoder"]

        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor
            y_z: torch.Tensor
            x_z: torch.Tensor
            # - encoder / classifier -
            z_x = encoder(x)
            y_z, z_x = classifier(z_x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")
            # - decoder -
            x_z = decoder(z_x)
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")

            # calculate loss
            classification_loss = torch.nn.functional.cross_entropy(y_z, y)
            reconstruction_loss = torch.nn.functional.mse_loss(x_z, x)
            l = 0.8
            minibatch_loss = l * classification_loss + (1 - l) * reconstruction_loss

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

    if min(stats) == avg_loss and not config.exc_dry_run:
        logging.info("checkpoint - saving current model and optimizer state")
        model_state_path = os.path.join(experiment_path, "model.pth")
        optim_state_path = os.path.join(experiment_path, "optim.pth")
        torch.save(nn.state_dict(), model_state_path)
        torch.save(optim.state_dict(), optim_state_path)


def test_model(
    nn: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
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
        encoder = nn["encoder"]
        classifier = nn["classifier"]
        decoder = nn["decoder"]

        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor
            y_z: torch.Tensor
            x_z: torch.Tensor
            # - encoder / classifier -
            z_x = encoder(x)
            y_z, z_x = classifier(z_x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")
            # - decoder -
            x_z = decoder(z_x[..., None, None])
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")

            # calculate loss
            classification_loss = torch.nn.functional.cross_entropy(y_z, y)
            reconstruction_loss = torch.nn.functional.mse_loss(x_z, x)
            l = 0.8
            minibatch_loss = l * classification_loss + (1 - l) * reconstruction_loss

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

            # accumulate plots
            if current_plot < img_count:
                ax[current_plot, 0].imshow(x.to("cpu")[0, 0])
                ax[current_plot, 1].imshow(x_z.to("cpu")[0, 0])
                current_plot += 1

    # post-testing
    avg_loss = sum_loss / size
    stats.append(avg_loss)
    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, f"test_e{epoch}.png"))
    plt.close()
