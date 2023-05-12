from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import compute_shapes, gather_samples, gen_epoch_acc, set_requires_grad
from .flow.util import decode_mask
from .resnet.residual_block import get_encoder


def train_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Tuple[torch.optim.Optimizer, ...],
    **kwargs,
) -> dict:
    return epoch_adv(model, epoch, config, optim)


def test_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    **kwargs,
) -> dict:
    return epoch_adv(model, epoch, config, None)


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.dataset_info is not None
    assert config.image_chw is not None

    c, h, w = config.image_chw
    (k0, k1, c0, c1, c2, cm, h2, w2, num_bins, num_labels) = compute_shapes(config)

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

    u_mask = torch.zeros(cm).bool()
    u_mask[::2] = True
    uI, uT = decode_mask(u_mask)
    rqs_coupling_args_uA = {
        "i_channels": uI,
        "t_channels": uT,
        "num_bins": num_bins,
    }
    rqs_coupling_args_uB = {
        "i_channels": uT,
        "t_channels": uI,
        "num_bins": num_bins,
    }
    model = torch.nn.ModuleDict(
        {
            # Base Distribution
            "dist": flow.distributions.StandardNormal(cm * h2 * w2),
            # MNIST: (cm * 4 * 4) <-> (cm, 4, 4) <-> (64, 4, 4) <-> (16, 8, 8) <-> (1, 32, 32)
            "generator": flow.Compose(
                [
                    #
                    flow.nn.Squeeze(factor=k0),
                    flow.nn.ActNorm(c1),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                    flow.nn.ActNorm(c1),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
                    flow.nn.ActNorm(c1),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                    flow.nn.ActNorm(c1),
                    #
                    flow.nn.Squeeze(factor=k1),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                    flow.nn.ActNorm(c2),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
                    flow.nn.ActNorm(c2),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                    flow.nn.ActNorm(c2),
                    #
                    flow.nn.Proj(c2, cm),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                    flow.nn.ActNorm(cm),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uB),
                    flow.nn.ActNorm(cm),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                    #
                    flow.nn.Flatten(input_shape=(cm, h2, w2)),
                    flow.nn.ActNorm(cm * h2 * w2),
                    flow.nn.Tanh(),  # to restrict the domain to (-1, +1)
                    #
                ]
            ),
            # MNIST: (1, 32, 32) -> (1,)
            "classifier": torch.nn.Sequential(
                get_encoder(
                    input_chw=config.image_chw,
                    num_features=num_bins,
                ),
            ),
        }
    )

    # set up optimizers
    optim_generator = torch.optim.AdamW(
        params=model["generator"].parameters(),
        lr=config.optim_lr,
    )
    optim_classifier = torch.optim.AdamW(
        params=model["classifier"].parameters(),
        lr=config.optim_lr,
    )

    return model, (optim_generator, optim_classifier)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:
    assert config.image_chw
    B, C, H, W = (config.batch_size, *config.image_chw)
    x_size = (B, C, H, W)
    c, h, w = C * 64, H // 8, W // 8
    cm = config.manifold_d // h // w
    uv_size = (B, c, h, w)
    u_size = (B, cm * h * w)
    torchinfo.summary(model["generator"], input_size=x_size, depth=5)
    # torchinfo.summary(model["x_flow"], input_size=x_size, depth=5)


def epoch_adv(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
    **kwargs,
) -> dict:
    assert config.train_loader is not None

    # initialize loop
    if optim:
        model.train()
        prefix = "TRN"
    else:
        model.eval()
        prefix = "TST"

    data_loader = config.train_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0

    # logic
    set_requires_grad(model, True)
    with torch.enable_grad():
        iterable = tqdm(data_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore
        generator: flow.FlowTransform = model["generator"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        u_pred = []
        v_pred = []
        z_pred = []
        z_nll = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)
            B = x.size(0)

            z_x, logabsdet_xz = generator(x, forward=True)
            x_z, logabsdet_zx = generator(z_x, forward=False)

            # calculate losses
            loss_z = dist.log_prob(z_x)
            loss_x = F.mse_loss(x_z, x)

            # accumulate predictions
            # u_pred.extend(u_x.detach().cpu().numpy())
            # v_pred.extend(v_x.detach().cpu().numpy())
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z_x.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y)

            # compute minibatch loss
            loss_minibatch = loss_x + loss_z

            # backward pass (if optimizer is given)
            if optim:
                (optim_generator, optim_classifier) = optim
                optim_generator.zero_grad()
                optim_classifier.zero_grad()
                loss_minibatch.backward()
                optim_classifier.step()
                optim_generator.step()

            # accumulate sum loss
            sum_loss += loss_minibatch.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{loss_minibatch.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-training
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(
        f"[{prefix}] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: [{acc_score[0]:.4f}, {acc_score[1]:.4f}, {acc_score[2]:.4f}]"
    )

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "u_pred": np.array(u_pred),
        "v_pred": np.array(v_pred),
        "z_pred": np.array(z_pred),
        "z_nll": np.array(z_nll),
        "samples": samples,
    }
