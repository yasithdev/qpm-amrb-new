import logging
import os
from typing import List, Tuple

import torch
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from .common import (
    generate_confusion_matrix,
    get_classifier,
    load_saved_state,
    save_state,
    set_requires_grad,
)
from .resnet import get_decoder, get_encoder


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:

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
        load_saved_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
            config=config,
        )

    return model, optim


def train_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
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
        encoder = model["encoder"]
        decoder = model["decoder"]
        classifier = model["classifier"]

        x: torch.Tensor
        y: torch.Tensor
        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor = encoder(x)
            y_z: torch.Tensor = classifier(z_x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")
            x_z: torch.Tensor = decoder(z_x)
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
        save_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
            config=config,
        )


def test_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> None:
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
        encoder = model["encoder"]
        decoder = model["decoder"]
        classifier = model["classifier"]

        x: torch.Tensor
        y: torch.Tensor

        y_true = []
        y_pred = []

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

            # accumulate predictions
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y_z, dim=1).cpu().numpy())

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
                ax[current_plot, 0].imshow(x.cpu().numpy()[0, 0])
                ax[current_plot, 1].imshow(x_z.cpu().numpy()[0, 0])
                current_plot += 1

    # post-testing
    avg_loss = sum_loss / size
    stats.append(avg_loss)
    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, f"test_e{epoch}.png"))
    plt.close()

    # save confusion matrix
    generate_confusion_matrix(
        y_pred=y_pred,
        y_true=y_true,
        labels=config.train_loader.dataset.labels,
        experiment_path=experiment_path,
        epoch=epoch,
    )
