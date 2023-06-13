import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from .common import gather_samples, get_classifier
from .resnet import get_decoder, get_encoder


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.dataset_info is not None
    assert config.image_chw is not None

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

    return model, (optim,)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:
    assert config.image_chw
    B, C, H, W = (config.batch_size, *config.image_chw)

    enc_in_size = (B, C, H, W)
    enc_out_size = (B, 128, 1, 1)
    cls_in_size = enc_out_size
    dec_in_size = enc_out_size

    torchinfo.summary(model["encoder"], input_size=enc_in_size, depth=5)
    torchinfo.summary(model["classifier"], input_size=cls_in_size, depth=5)
    torchinfo.summary(model["decoder"], input_size=dec_in_size, depth=5)


def step_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]],
    **kwargs,
) -> dict:

    # pre-step
    if optim:
        data_loader = config.train_loader
        assert data_loader
        model.train()
        prefix = "TRN"
    else:
        data_loader = config.test_loader
        assert data_loader
        model.eval()
        prefix = "TST"

    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0

    # step
    with torch.enable_grad():
        iterable = tqdm(
            data_loader, desc=f"[{prefix}] Epoch {epoch}", **config.tqdm_args
        )
        encoder = model["encoder"]
        decoder = model["decoder"]
        classifier = model["classifier"]

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        y_nll = []
        z_norm = []
        samples = []

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

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_z.detach().softmax(-1).cpu().numpy())
            z_norm.extend(z_x.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # calculate loss
            classification_loss = F.cross_entropy(y_z, y, label_smoothing=0.1)
            reconstruction_loss = F.mse_loss(x_z, x)
            l = 0.8
            minibatch_loss = l * classification_loss + (1 - l) * reconstruction_loss

            # backward pass
            if optim:
                optim[0].zero_grad()
                minibatch_loss.backward()
                optim[0].step()

            # save nll
            nll = F.nll_loss(y_z.log_softmax(-1), y.argmax(-1), reduction="none")
            y_nll.extend(nll.detach().cpu().numpy())

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-step
    avg_loss = sum_loss / size

    return {
        "loss": avg_loss,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_nll": np.array(y_nll),
        "z_norm": np.array(z_norm),
        "samples": samples,
    }
