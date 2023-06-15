import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from .capsnet.caps import ConvCaps2D, FlattenCaps, LinearCapsDR, squash
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .common import (
    Functional,
    edl_loss,
    edl_probs,
    gather_samples,
    get_conv_out_shape,
    margin_loss,
)
from .resnet import get_decoder


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.dataset_info is not None
    assert config.image_chw is not None

    num_labels = config.dataset_info["num_train_labels"]
    kernel_conv = (9, 9)
    kernel_caps = (3, 3)
    stride_conv = 1
    stride_caps = 2

    # compute output shapes at each layer
    (c, h, w) = config.image_chw
    (c1, d1, h1, w1) = 8, 16, h // stride_conv, w // stride_conv
    (c2, d2, h2, w2) = 8, 32, h1 // stride_caps, w1 // stride_caps
    (c3, d3, h3, w3) = 8, 32, h2 // stride_caps, w2 // stride_caps
    (c4, d4, h4, w4) = 8, 32, h3 // stride_caps, w3 // stride_caps

    manifold_d = config.manifold_d

    # (B,C,H,W)->(B,D,K)
    encoder = torch.nn.Sequential(
        # (B,C,H,W)->(B,c1*d1,h1,w1)
        torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=c,
                out_channels=c1 * d1,
                kernel_size=kernel_conv,
                stride=stride_conv,
            ),
            # (B,c1*d1,h1,w1)->(B,c1,d1,h1,w1)
            Functional(partial(conv_to_caps, out_capsules=(c1, d1))),
            # Functional(squash),
            torch.nn.GroupNorm(1, c1),
        ),
        # (B,c1,d1,h1,w1)->(B,c2,d2,h2,w2)
        torch.nn.Sequential(
            ConvCaps2D(
                in_capsules=(c1, d1),
                out_capsules=(c2, d2),
                kernel_size=kernel_caps,
                stride=stride_caps,
            ),
            # Functional(squash),
            torch.nn.GroupNorm(1, c2),
        ),
        ConvCaps2D(
            in_capsules=(c2, d2),
            out_capsules=(c3, d3),
            kernel_size=kernel_caps,
            stride=stride_caps,
        ),
        # Functional(squash),
        torch.nn.GroupNorm(num_groups=1, num_channels=caps_cd_1[0]),
        ConvCaps2D(
            in_capsules=caps_cd_1,
            out_capsules=caps_cd_2,
            kernel_size=caps_kernel_hw,
            stride=caps_stride,
        ),
        # Functional(squash),
        torch.nn.GroupNorm(num_groups=1, num_channels=caps_cd_2[0]),
        FlattenCaps(),
        LinearCapsDR(
            in_capsules=(caps_cd_2[0], caps_cd_2[1] * h4 * w4),
            out_capsules=(manifold_d, num_labels),
        ),
    )

    # (B,D,K)->[(B,K),(B,D)]
    classifier = MaskCaps()

    decoder = get_decoder(
        num_features=manifold_d,
        output_chw=config.image_chw,
    )

    model = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "classifier": classifier,
            "decoder": decoder,
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

    assert config.dataset_info
    assert config.image_chw

    B = config.batch_size
    D = config.manifold_d
    K = config.dataset_info["num_train_labels"]
    (C, H, W) = config.image_chw

    torchinfo.summary(model["encoder"], input_size=(B, C, H, W), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, D, K), depth=5)
    torchinfo.summary(model["decoder"], input_size=(B, D, 1, 1), depth=5)


def step_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
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
        classifier = model["classifier"]
        decoder = model["decoder"]

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        y_ucty = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            z_x: torch.Tensor
            y_z: torch.Tensor
            x_z: torch.Tensor

            # encoder
            z_x = encoder(x)
            logging.debug(f"encoder: ({x.size()}) -> ({z_x.size()})")

            # classifier
            y_z, z_x = classifier(z_x)
            pY, uY = edl_probs(y_z.detach())

            # decoder
            x_z = decoder(z_x[..., None, None])
            logging.debug(f"decoder: ({z_x.size()}) -> ({x_z.size()})")

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(pY.detach().cpu().numpy())
            y_ucty.extend(uY.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # calculate loss
            # L_y_z = margin_loss(y_z, y) - replaced with evidential loss
            L_y_z = edl_loss(y_z, y, epoch)
            mask = pY.argmax(-1).eq(y.argmax(-1)).nonzero()
            L_x_z = (x_z[mask] - x[mask]).pow(2).flatten(1).mean(-1)
            l = 0.999
            L_minibatch = (l * L_y_z + (1 - l) * L_x_z).mean()

            # backward pass
            if optim:
                (optim_gd,) = optim
                optim_gd.zero_grad()
                L_minibatch.backward()
                optim_gd.step()

            # accumulate sum loss
            sum_loss += L_minibatch.item() * config.batch_size

            # logging
            log_stats = {
                "Loss(mb)": f"{L_minibatch:.4f}",
                "Loss(x)": f"{L_x_z.mean():.4f}",
                "Loss(y)": f"{L_y_z.mean():.4f}",
            }
            iterable.set_postfix(log_stats)

    # post-step
    avg_loss = sum_loss / size

    return {
        "loss": avg_loss,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_ucty": np.array(y_ucty),
        "samples": samples,
    }
