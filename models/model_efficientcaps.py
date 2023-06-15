import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from .capsnet.caps import ConvCaps2D, FlattenCaps
from .capsnet.common import conv_to_caps
from .capsnet.deepcaps import MaskCaps
from .capsnet.efficientcaps import LinearCapsAR, squash
from .common import Functional, gather_samples, get_conv_out_shape, margin_loss
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
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.dataset_info is not None
    assert config.image_chw is not None

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
        LinearCapsAR(
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
    optim = torch.optim.AdamW(**optim_config)

    return model, (optim,)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:

    assert config.dataset_info
    assert config.image_chw

    B, C, H, W = (config.batch_size, *config.image_chw)
    enc_in_size = (B, C, H, W)

    c, n = 16, config.dataset_info["num_train_labels"]
    enc_out_size = (B, c, n)

    cls_in_size = enc_out_size
    dec_in_size = (B, c, 1, 1)

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
        classifier = model["classifier"]
        decoder = model["decoder"]

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
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_z.detach().softmax(-1).cpu().numpy())
            z_norm.extend(z_x.detach().flatten(start_dim=1).norm(2, -1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # calculate loss
            classification_loss = margin_loss(y_z, y)
            mask = y_z.argmax(-1).eq(y.argmax(-1)).nonzero()
            reconstruction_loss = F.mse_loss(x_z[mask], x[mask])
            l = 0.9
            minibatch_loss = l * classification_loss + (1 - l) * reconstruction_loss

            # backward pass
            if optim:
                (optim_gd,) = optim
                optim_gd.zero_grad()
                minibatch_loss.backward()
                optim_gd.step()

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
        "z_norm": np.array(z_norm),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_nll": np.array(y_nll),
        "samples": samples,
    }
