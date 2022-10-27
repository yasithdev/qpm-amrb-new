import logging
import os
from functools import partial
from typing import List, Tuple

import torch
import torch.utils.data
from config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

from .capsnet.caps import ConvCaps2D, FlattenCaps, LinearCapsDR
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .capsnet.efficientcaps import squash
from .common import (
    Functional,
    gen_epoch_stats,
    get_conv_out_shape,
    load_saved_state,
    plot_confusion_matrix,
    save_state,
    set_requires_grad,
)
from .resnet import get_decoder

caps_cd_1 = (8, 16)
caps_cd_2 = (8, 32)
conv_kernel_hw = (9, 9)
caps_kernel_hw = (3, 3)
conv_stride = 1
caps_stride = 2
out_caps_c = 16


def load_model_and_optimizer(
    config: Config,
    experiment_path: str,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:

    out_caps_d = config.dataset_info["num_train_labels"]

    # compute hw shapes of conv
    (h0, w0) = config.image_chw[1:]
    (conv_kh, conv_kw) = conv_kernel_hw
    (caps_kh, caps_kw) = caps_kernel_hw
    (h1, w1) = get_conv_out_shape(h0, conv_kh, conv_stride), get_conv_out_shape(
        w0, conv_kw, conv_stride
    )
    (h4, w4) = get_conv_out_shape(
        h1, caps_kh, caps_stride, blocks=3
    ), get_conv_out_shape(w1, caps_kw, caps_stride, blocks=3)

    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=config.image_chw[0],
            out_channels=caps_cd_1[0] * caps_cd_1[1],
            kernel_size=conv_kernel_hw,
            stride=conv_stride,
        ),
        Functional(partial(conv_to_caps, out_capsules=caps_cd_1)),
        Functional(squash),
        ConvCaps2D(
            in_capsules=caps_cd_1,
            out_capsules=caps_cd_1,
            kernel_size=caps_kernel_hw,
            stride=caps_stride,
        ),
        Functional(squash),
        ConvCaps2D(
            in_capsules=caps_cd_1,
            out_capsules=caps_cd_1,
            kernel_size=caps_kernel_hw,
            stride=caps_stride,
        ),
        Functional(squash),
        ConvCaps2D(
            in_capsules=caps_cd_1,
            out_capsules=caps_cd_2,
            kernel_size=caps_kernel_hw,
            stride=caps_stride,
        ),
        Functional(squash),
        FlattenCaps(),
        LinearCapsDR(
            in_capsules=(caps_cd_2[0], caps_cd_2[1] * h4 * w4),
            out_capsules=(out_caps_c, out_caps_d),
        ),
        Functional(squash),
    )

    classifier = MaskCaps()

    decoder = get_decoder(
        num_features=out_caps_c,
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
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    stats: List,
    experiment_path: str,
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
        encoder = model["encoder"]
        classifier = model["classifier"]
        decoder = model["decoder"]

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []

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

            # accumulate predictions
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y_z, dim=1).cpu().numpy())

            # calculate loss
            classification_loss = torch.nn.functional.cross_entropy(y_z, y)
            reconstruction_loss = torch.nn.functional.mse_loss(x_z, x)
            l = 0.8
            minibatch_loss = l * classification_loss + (1 - l) * reconstruction_loss

            # backward pass
            optim.zero_grad()
            minibatch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    stats: List,
    experiment_path: str,
    **kwargs,
) -> dict:

    # initialize loop
    model = model.float().to(config.device)
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
        classifier = model["classifier"]
        decoder = model["decoder"]

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []

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
    cf_matrix, acc_score = gen_epoch_stats(y_pred=y_pred, y_true=y_true)
    stats.append(avg_loss)

    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    # save generated plot
    if not config.exc_dry_run:
        plt.savefig(os.path.join(experiment_path, f"test_e{epoch}.png"))
    plt.close()

    # save confusion matrix
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
