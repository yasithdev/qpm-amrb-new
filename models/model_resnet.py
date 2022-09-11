import logging
import os
from typing import List

import torch
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from .common import get_classifier
from .resnet import get_decoder, get_encoder
from .util import set_requires_grad


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
):

    num_features = 128

    # BCHW -> BD11
    encoder = get_encoder(
        input_chw=config.image_chw,
        num_features=num_features,
    )

    # BD11 -> BCHW
    decoder = get_decoder(
        num_features=num_features,
        output_chw=config.image_chw,
    )

    # BD11 -> BL
    num_labels = config.dataset_info["num_train_labels"]
    classifier = get_classifier(
        in_features=num_features,
        out_features=num_labels,
    )

    model = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "decoder": decoder,
            "classifier": classifier,
        }
    )

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
        decoder = nn["decoder"]
        classifier = nn["classifier"]

        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor = encoder(x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")
            x_z: torch.Tensor = decoder(z_x)
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")
            y_z: torch.Tensor = classifier(z_x)

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
        decoder = nn["decoder"]
        classifier = nn["classifier"]

        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor = encoder(x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")
            x_z: torch.Tensor = decoder(z_x)
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")
            y_z: torch.Tensor = classifier(z_x)

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
