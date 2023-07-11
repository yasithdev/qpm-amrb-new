import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from config import Config


from .resnet import ResidualBlock
from .capsnet.caps import FlattenCaps, LinearCapsDR
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .common import Functional, edl_loss, edl_probs, gather_samples
from .resnet import get_decoder


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:
    assert config.image_chw is not None

    ind_targets = config.get_ind_labels()
    num_labels = len(ind_targets)

    # compute hw shapes of conv
    (C, H, W) = config.image_chw
    manifold_d = config.manifold_d

    # (B,C,H,W)->(B,D,K)
    encoder = torch.nn.Sequential(
        # (B,C,H,W)->(B,32,H/2,W/2)
        ResidualBlock(
            in_channels=C,
            hidden_channels=C * 4,
            out_channels=C * 4,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B,32,H/2,W/2) -> (B,64,H/4,W/4)
        ResidualBlock(
            in_channels=C * 4,
            hidden_channels=C * 8,
            out_channels=C * 8,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        # (B,64,H/4,W/4) -> (B,128,H/8,W/8)
        ResidualBlock(
            in_channels=C * 8,
            hidden_channels=C * 16,
            out_channels=C * 16,
            stride=2,
            conv=torch.nn.Conv2d,
        ),
        Functional(partial(conv_to_caps, out_capsules=(16, C))),
        FlattenCaps(),
        LinearCapsDR(
            in_capsules=(16, C * H * W // 64),
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
    assert config.image_chw is not None

    ind_targets = config.get_ind_labels()
    B = config.batch_size
    D = config.manifold_d
    K = len(ind_targets)
    (C, H, W) = config.image_chw

    torchinfo.summary(model["encoder"], input_size=(B, C, H, W), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, D, K), depth=5)
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
    if prefix == "TRN":
        data_loader = config.train_loader
        assert data_loader
        model.train()
    elif prefix == "TST":
        data_loader = config.test_loader
        assert data_loader
        model.eval()
    elif prefix == "OOD":
        data_loader = config.ood_loader
        assert data_loader
        model.eval()
    else:
        raise ValueError()

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
            l = 0.8
            L_minibatch = (l * L_y_z + (1 - l) * L_x_z).mean()

            # backward pass
            if prefix == "TRN":
                assert optim is not None
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
