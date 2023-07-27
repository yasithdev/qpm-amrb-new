from collections import defaultdict
import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from config import Config

from .capsnet.caps import ConvCaps2D, FlattenCaps
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .capsnet.efficientcaps import LinearCapsAR, squash
from .common import Functional, edl_loss, edl_probs, gather_samples, get_conv_out_shape
from .resnet import get_decoder

caps_cd_1 = (8, 16)
caps_cd_2 = (8, 32)
conv_kernel_hw = (9, 9)
caps_kernel_hw = (3, 3)
conv_stride = 1
caps_stride = 2


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:
    assert config.image_chw is not None

    ind_targets = config.get_ind_labels()
    num_labels = len(ind_targets)
    S = get_conv_out_shape

    # compute hw shapes of conv
    (h0, w0) = config.image_chw[1:]
    (conv_kh, conv_kw) = conv_kernel_hw
    (caps_kh, caps_kw) = caps_kernel_hw
    (h1, w1) = S(h0, conv_kh, conv_stride), S(w0, conv_kw, conv_stride)
    (h4, w4) = S(h1, caps_kh, caps_stride, n=3), S(w1, caps_kw, caps_stride, n=3)
    manifold_d = config.manifold_d

    # (B,C,H,W)->(B,D,K)
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
        LinearCapsAR(
            in_capsules=(caps_cd_2[0], caps_cd_2[1] * h4 * w4),
            out_capsules=(manifold_d, num_labels),
        ),
        Functional(squash),
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
        classifier = model["classifier"]
        decoder = model["decoder"]

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
            y_z, z_x = classifier(z_x)
            pY, uY = edl_probs(y_z.detach())

            # decoder
            x_z = decoder(z_x[..., None, None])
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
            l = 0.999
            L_minibatch = (l * L_y_z + (1 - l) * L_x_z).mean()
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
