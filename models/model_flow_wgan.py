from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config

from . import flow
from .common import (
    gather_samples,
    gen_epoch_acc,
    set_requires_grad,
    compute_shapes,
    GradientHook,
)
from .resnet.residual_block import get_encoder
from .flow.util import decode_mask

import numpy as np


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
            # MNIST: (cm * 4 * 4) -> (cm, 4, 4)
            "u_flow": flow.Compose(
                [
                    #
                    flow.nn.Unflatten(target_shape=(cm, h2, w2)),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                    flow.nn.ActNorm(cm),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uB),
                    flow.nn.ActNorm(cm),
                ]
            ),
            # MNIST: (64, 4, 4) -> (16, 8, 8) -> (1, 32, 32)
            "x_flow": flow.Compose(
                [
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
                    flow.nn.ActNorm(c2),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                    flow.nn.ActNorm(c2),
                    #
                    flow.nn.Unsqueeze(factor=k1),
                    #
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
                    flow.nn.ActNorm(c1),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                    #
                    flow.nn.Unsqueeze(factor=k0),
                    #
                ]
            ),
            # MNIST: (1, 32, 32) -> (num_labels,)
            "classifier": torch.nn.Sequential(
                get_encoder(
                    input_chw=config.image_chw,
                    num_features=num_labels,
                ),
            ),
            # MNIST: (1, 32, 32) -> (1,)
            "critic": torch.nn.Sequential(
                get_encoder(
                    input_chw=config.image_chw,
                    num_features=1,
                ),
            ),
        }
    )

    hook_grad = GradientHook(torch.nn.ModuleList([model["u_flow"], model["x_flow"]]))
    hook_grad.set_negate_grads_hook()

    # set up optimizer
    optim_g = torch.optim.AdamW(
        params=[*model["x_flow"].parameters(), *model["u_flow"].parameters()],
        lr=config.optim_lr,
    )
    optim_d = torch.optim.AdamW(
        params=[*model["critic"].parameters(), *model["classifier"].parameters()],
        lr=config.optim_lr,
    )

    return model, (optim_g, optim_d)


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
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        u_flow: flow.FlowTransform = model["u_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore
        critic: torch.nn.Module = model["critic"]  # type: ignore
        target_real = torch.ones(config.batch_size, 1).to(config.device)
        target_fake = torch.zeros(config.batch_size, 1).to(config.device)
        target = torch.cat([target_real, target_fake], dim=0)

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

            # target image classification
            y_x = classifier(x)
            y_x = F.gumbel_softmax(y_x)

            # image generation
            z = dist.sample(B).to(x.device)
            # z = torch.concat([z, y_x.detach()], dim=1)
            u_z: torch.Tensor
            u_z, logabsdet_zu = u_flow(z, forward=True)
            v_z = u_z.new_zeros(B, c2 - cm, h2, w2)
            uv_z = flow.nn.join(u_z, v_z)
            x_z, logabsdet_ux = x_flow(uv_z, forward=True)

            # generated image classification
            y_z = classifier(x_z)
            y_z = F.gumbel_softmax(y_z)

            # critic
            pred: torch.Tensor = critic(torch.cat([x, x_z], dim=0))

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_x.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y_x)

            # calculate losses
            loss_minibatch = torch.mean(pred * target)

            # backward pass (if optimizer is given)
            if optim:
                (optim_g, optim_d) = optim
                optim_d.zero_grad()
                optim_g.zero_grad()
                loss_minibatch.backward()
                optim_d.step()
                optim_g.step()

            # weight clipping for lipschitz constraint
            for param in critic.parameters():
                param.data.clamp_(-0.01, 0.01)

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
