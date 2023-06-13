from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import compute_flow_shapes, gather_samples
from .flow.util import decode_mask
from .resnet.residual_block import get_encoder


def compute_sizes(config: Config) -> dict:
    assert config.image_chw
    C, H, W = config.image_chw
    (_, _, _, _, c2, cm, h2, w2, _, _) = compute_flow_shapes(config)

    return {
        "x": (C, H, W),
        "uv": (c2 * h2 * w2),
        "u": (cm * h2 * w2),
        "v": ((c2 - cm) * h2 * w2),
    }


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.image_chw
    (k0, k1, _, c1, c2, cm, h2, w2, num_bins, num_labels) = compute_flow_shapes(config)

    x1_mask = torch.zeros(c1).bool()
    x1_mask[::2] = True
    x1I, x1T = decode_mask(x1_mask)
    rqs_coupling_args_x1A = {
        "i_channels": x1I,
        "t_channels": x1T,
        "num_bins": num_bins,
    }
    rqs_coupling_args_x1B = {
        "i_channels": x1T,
        "t_channels": x1I,
        "num_bins": num_bins,
    }

    x2_mask = torch.zeros(c2).bool()
    x2_mask[::2] = True
    x2I, x2T = decode_mask(x2_mask)
    rqs_coupling_args_x2A = {
        "i_channels": x2I,
        "t_channels": x2T,
        "num_bins": num_bins,
    }
    rqs_coupling_args_x2B = {
        "i_channels": x2T,
        "t_channels": x2I,
        "num_bins": num_bins,
    }

    # Base Distribution
    dist_z = flow.distributions.StandardNormal(k=c2 * h2 * w2, mu=0.0, std=1.0)

    # MNIST: (1, 32, 32) <-> (16, 8, 8) <-> [RQS] <-> (64, 4, 4) <-> [RQS] <-> (784)
    flow_x = flow.Compose(
        [
            #
            flow.nn.Tanh(),
            flow.nn.Squeeze(factor=k0),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
            flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
            #
            flow.nn.Tanh(),
            flow.nn.Squeeze(factor=k1),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
            flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
            #
            flow.nn.Tanh(),
            flow.nn.Flatten(input_shape=(c2, h2, w2)),
            #
        ]
    )

    # MNIST: (1, 32, 32) -> (K,)
    classifier = torch.nn.Sequential(
        get_encoder(
            input_chw=config.image_chw,
            num_features=num_labels,
        ),
    )

    model = torch.nn.ModuleDict(
        {
            "dist_z": dist_z,
            "flow_x": flow_x,
            "classifier": classifier,
        }
    )

    # set up optimizer
    optim_g = torch.optim.AdamW(
        params=flow_x.parameters(),
        lr=config.optim_lr,
    )
    optim_d = torch.optim.AdamW(
        params=classifier.parameters(),
        lr=config.optim_lr,
    )

    return model, (optim_g, optim_d)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:
    B = config.batch_size
    sizes = compute_sizes(config)
    torchinfo.summary(model["flow_x"], input_size=(B, *sizes["x"]), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, *sizes["x"]), depth=5)


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
    sizes = compute_sizes(config)

    # step
    with torch.enable_grad():
        iterable = tqdm(
            data_loader, desc=f"[{prefix}] Epoch {epoch}", **config.tqdm_args
        )
        dist_z: flow.distributions.Distribution = model["dist_z"]  # type: ignore
        flow_x: flow.Compose = model["flow_x"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        x_z: torch.Tensor

        y_true = []
        y_pred = []
        z_pred = []
        z_nll = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)
            B = x.size(0)

            # x -> (u x v) -> u -> z
            z_x, _ = flow_x(x, forward=True)

            # u -> (u x 0) -> x~
            x_z, _ = flow_x(z_x, forward=False)

            # classifier
            y_z = classifier(x_z)

            # calculate losses
            loss_z = -dist_z.log_prob(z_x)
            loss_y = F.cross_entropy(y_z, y)

            # accumulate predictions
            z_pred.extend(z_x.detach().flatten(start_dim=1).cpu().numpy())
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_z.detach().softmax(-1).cpu().numpy())
            z_nll.extend(loss_z.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # compute minibatch loss
            loss_minibatch = loss_z.mean() + loss_y

            # backward pass (if optimizer is given)
            if optim:
                (optim_g, optim_d) = optim
                optim_g.zero_grad()
                optim_d.zero_grad()
                loss_minibatch.backward()
                optim_d.step()
                optim_g.step()

            # accumulate sum loss
            sum_loss += loss_minibatch.item() * config.batch_size

            # logging
            log_stats = {
                "Loss(mb)": f"{loss_minibatch:.4f}",
                "Loss(z)": f"{loss_z.mean():.4f}",
                "Loss(y)": f"{loss_y:.4f}",
            }
            iterable.set_postfix(log_stats)

    # post-step
    avg_loss = sum_loss / size

    return {
        "loss": avg_loss,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_ucty": np.array(y_ucty),
        "z_nll": np.array(z_nll),
        "u_norm": np.array(u_norm),
        "v_norm": np.array(v_norm),
        "z_norm": np.array(z_norm),
        "samples": samples,
    }
