from collections import defaultdict
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from config import Config

from .common import gather_samples, edl_probs, edl_loss
from .resnet import get_decoder, get_encoder


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:
    assert config.image_chw is not None

    ind_targets = config.get_ind_labels()
    num_labels = len(ind_targets)

    # (B, C, H, W) -> (B, D, 1, 1)
    encoder = get_encoder(
        input_chw=config.image_chw,
        num_features=config.manifold_d,
    )

    # (B, D, 1, 1) -> (B, C, H, W)
    decoder = get_decoder(
        num_features=config.manifold_d,
        output_chw=config.image_chw,
    )

    # (B, D, 1, 1) -> (B, K)
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(config.manifold_d, num_labels),
    )

    model = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "decoder": decoder,
            "classifier": classifier,
        }
    )

    # set up optimizer
    optim_config = {"params": model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.AdamW(**optim_config)

    return model, (optim,)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:
    assert config.image_chw is not None

    B = config.batch_size
    D = config.manifold_d
    (C, H, W) = config.image_chw

    torchinfo.summary(model["encoder"], input_size=(B, C, H, W), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, D, 1, 1), depth=5)
    torchinfo.summary(model["decoder"], input_size=(B, D, 1, 1), depth=5)


def step_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    prefix: str,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
    **kwargs,
) -> dict:
    # pre-step
    if prefix == "train":
        data_loader = config.train_loader
        assert data_loader is not None
        model.train()
    elif prefix == "test_ind":
        data_loader = config.test_loader
        assert data_loader is not None
        model.eval()
    elif prefix == "test_ood":
        data_loader = config.ood_loader
        assert data_loader is not None
        model.eval()
    else:
        raise ValueError()

    size = 0
    sum_loss = defaultdict(lambda: 0.)

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
        y_prob = []
        y_ucty = []
        samples = []

        for x, y in iterable:
            # cast x and y to float
            x = x.float().to(config.device)
            y = y.long().to(config.device)
            B = x.size(0)

            # forward pass
            z_x: torch.Tensor
            y_z: torch.Tensor
            x_z: torch.Tensor

            # encoder
            z_x = encoder(x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")

            # classifier
            y_z = classifier(z_x)
            pY, uY = edl_probs(y_z.detach())

            # decoder
            x_z = decoder(z_x)
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")

            # accumulate predictions
            y_true.extend(y.detach().cpu().numpy())
            y_prob.extend(pY.detach().cpu().numpy())
            y_ucty.extend(uY.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # calculate loss
            # L_y_z = margin_loss(y_z, y) - replaced with evidential loss
            L_y_z = edl_loss(y_z, y, epoch)
            L_x_z = (x_z - x).pow(2).flatten(1).mean(-1)
            l = 1.0
            L_minibatch = (L_y_z + l * L_x_z).mean()
            L_y_minibatch = L_y_z.mean()
            L_x_minibatch = L_x_z.mean()

            # backward pass
            if prefix == "train":
                assert optim is not None
                (optim_gd,) = optim
                optim_gd.zero_grad()
                L_minibatch.backward()
                optim_gd.step()

            # accumulate sum loss
            sum_loss["agg"] += L_minibatch.item() * B
            sum_loss["y"] += L_y_minibatch.item() * B
            sum_loss["x"] += L_x_minibatch.item() * B
            size += B

            # logging
            log_stats = {
                "Loss(agg)": f"{L_minibatch:.4f}",
                "Loss(y)": f"{L_y_minibatch:.4f}",
                "Loss(x)": f"{L_x_minibatch:.4f}",
            }
            iterable.set_postfix(log_stats)

    # post-step
    avg_loss = {k: v / max(size, 1) for k, v in sum_loss.items()}

    return {
        "loss": avg_loss,
        "y_true": np.array(y_true),
        "y_prob": np.array(y_prob),
        "y_ucty": np.array(y_ucty),
        "samples": samples,
    }
