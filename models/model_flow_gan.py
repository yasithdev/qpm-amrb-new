from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import (
    GradientScaler,
    compute_shapes,
    gather_samples,
    gen_epoch_acc,
    get_gradient_ratios,
    set_requires_grad,
)
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
            # MNIST: (cm * 4 * 4) -> (cm, 4, 4) -> (64, 4, 4) -> (16, 8, 8) -> (1, 32, 32)
            "flow": flow.Compose(
                [
                    #
                    flow.nn.Unflatten(target_shape=(cm, h2, w2)),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                    flow.nn.ActNorm(cm),
                    flow.nn.LeakyReLU(),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uB),
                    flow.nn.ActNorm(cm),
                    flow.nn.LeakyReLU(),
                    #
                    flow.nn.Pad(c2 - cm),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
                    flow.nn.ActNorm(c2),
                    flow.nn.LeakyReLU(),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                    flow.nn.ActNorm(c2),
                    flow.nn.LeakyReLU(),
                    #
                    flow.nn.Unsqueeze(factor=k1),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
                    flow.nn.ActNorm(c1),
                    flow.nn.LeakyReLU(),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                    #
                    flow.nn.Unsqueeze(factor=k0),
                    #
                ]
            ),
            # MNIST: (1, 32, 32) -> (1,)
            "discriminator": torch.nn.Sequential(
                get_encoder(
                    input_chw=config.image_chw,
                    num_features=1,
                ),
            ),
        }
    )

    # set up optimizer
    optim_g = torch.optim.AdamW(
        params=model["flow"].parameters(),
        lr=config.optim_lr,
    )
    optim_d = torch.optim.AdamW(
        params=model["discriminator"].parameters(),
        lr=config.optim_lr * 0.1,
    )

    return model, (optim_g, optim_d)


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
    c2, cm, h2, w2 = compute_shapes(config)[4:8]

    # logic
    set_requires_grad(model, True)
    with torch.enable_grad():
        iterable = tqdm(data_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore
        flow: flow.FlowTransform = model["flow"]  # type: ignore
        discriminator: torch.nn.Module = model["discriminator"]  # type: ignore
        scaler = GradientScaler.apply
        bce = F.binary_cross_entropy_with_logits

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

            target_real: torch.Tensor = x.new_ones(B, 1)
            target_fake: torch.Tensor = x.new_zeros(B, 1)

            # image generation
            x_z: torch.Tensor
            z: torch.Tensor
            logabsdet_zx: torch.Tensor
            z = dist.sample(B).to(x.device)
            x_z, logabsdet_zx = flow(z, forward=True)
            # x_z = scaler(x_z)  # type: ignore

            z_x, logabsdet_xz = flow(x, forward=False)

            # discriminator
            pred_real: torch.Tensor = discriminator(x)
            pred_fake: torch.Tensor = discriminator(x_z)

            # accumulate predictions
            # u_pred.extend(u_x.detach().cpu().numpy())
            # v_pred.extend(v_x.detach().cpu().numpy())
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y)

            # calculate losses
            # loss_nll = dist.log_prob(z) - logabsdet_zu - logabsdet_ux
            # z_nll.extend(loss_nll.detach().cpu().numpy())
            # loss_mse = F.mse_loss(x_z, x)
            # loss_v = F.l1_loss(v, torch.zeros_like(v))
            # loss_real_real = bce(pred_real, target_real) + F.cross_entropy(y_x, y)

            # # loss wrt classifying real as real
            # loss_real_real = bce(pred_real, target_real)
            # # loss wrt classifying fake as fake
            # loss_fake_fake = bce(pred_fake, target_fake, reduction="none")
            # # loss wrt classifying fake as real
            # loss_fake_real = bce(pred_fake, target_real, reduction="none")

            # # total discriminator loss (for logging only)
            # # loss_discriminator = loss_real_real + torch.mean(loss_fake_fake)
            # # loss_generator = torch.mean(loss_fake_real)

            # gamma = get_gradient_ratios(loss_fake_real, loss_fake_fake, pred_fake)
            # GradientScaler.factor = gamma
            # loss_fake_scaled = (loss_fake_fake - loss_fake_real) / (1.0 - gamma)

            # compute minibatch loss
            loss_minibatch = F.mse_loss(z_x, torch.zeros_like(z_x)) + F.mse_loss(
                x, flow(z_x, forward=True)
            )

            # backward pass (if optimizer is given)
            if optim:
                (optim_g, optim_d) = optim
                optim_d.zero_grad()
                optim_g.zero_grad()
                loss_minibatch.backward()
                optim_d.step()
                optim_g.step()

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
