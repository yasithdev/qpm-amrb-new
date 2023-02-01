from typing import Tuple

import torch
from tqdm import tqdm

from config import Config

from . import flow
from .common import gather_samples, gen_epoch_acc, set_requires_grad
from .flow.util import decode_mask

import numpy as np


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:
    
    assert config.dataset_info is not None
    assert config.image_chw is not None

    # ambient (x) flow configuration
    k0, k1, k2, k3 = 2, 2, 2, 2
    c0, h0, w0 = config.image_chw
    c1, h1, w1 = c0 * k0 * k0, h0 // k0, w0 // k0
    c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1
    c3, h3, w3 = c2 * k2 * k2, h2 // k2, w2 // k2
    c4, h4, w4 = c3 * k3 * k3, h3 // k3, w3 // k3

    # manifold (m) flow configuration
    cm = config.manifold_c
    num_conv_layers = 2
    num_bins = 5

    channel_mask = torch.zeros(cm).bool()
    channel_mask[::2] = True
    i_channels, t_channels = decode_mask(channel_mask)

    affine_coupling_args_A = {
        "i_channels": i_channels,
        "t_channels": t_channels,
        "num_layers": num_conv_layers,
    }
    affine_coupling_args_B = {
        "i_channels": t_channels,
        "t_channels": i_channels,
        "num_layers": num_conv_layers,
    }
    rqs_coupling_args_A = {
        "i_channels": i_channels,
        "t_channels": t_channels,
        "num_bins": num_bins,
    }
    rqs_coupling_args_B = {
        "i_channels": t_channels,
        "t_channels": i_channels,
        "num_bins": num_bins,
    }
    model = torch.nn.ModuleDict(
        {
            # MNIST: (1, 32, 32) -> (4, 16, 16) -> (16, 8, 8) -> (64, 4, 4) -> (256, 2, 2)
            "x_flow": flow.Compose([
                flow.nn.ActNorm(c0),
                flow.nn.Squeeze(factor=k0),
                flow.nn.PiecewiseConformalConv2D(c1),
                flow.nn.ActNorm(c1),
                flow.nn.PiecewiseConformalConv2D(c1),
                flow.nn.ActNorm(c1),
                flow.nn.Squeeze(factor=k1),
                flow.nn.PiecewiseConformalConv2D(c2),
                flow.nn.ActNorm(c2),
                flow.nn.PiecewiseConformalConv2D(c2),
                flow.nn.ActNorm(c2),
                flow.nn.Squeeze(factor=k2),
                flow.nn.PiecewiseConformalConv2D(c3),
                flow.nn.ActNorm(c3),
                flow.nn.PiecewiseConformalConv2D(c3),
                flow.nn.ActNorm(c3),
                flow.nn.Squeeze(factor=k3),
                flow.nn.PiecewiseConformalConv2D(c4),
                flow.nn.ActNorm(c4),
                flow.nn.PiecewiseConformalConv2D(c4),
                flow.nn.Projection(c4, cm),
            ]),
            # MNIST: (cm, 2, 2) -> (cm, 2, 2)
            # affine coupling flow
            # "m_flow": flow.Compose([
            #     flow.nn.ActNorm(cm),
            #     flow.nn.PiecewiseConformalConv2D(cm),
            #     flow.nn.AffineCoupling(**affine_coupling_args_A),
            #     flow.nn.ActNorm(cm),
            #     flow.nn.PiecewiseConformalConv2D(cm),
            #     flow.nn.AffineCoupling(**affine_coupling_args_B),
            #     flow.nn.ActNorm(cm),
            #     flow.nn.PiecewiseConformalConv2D(cm),
            #     flow.nn.AffineCoupling(**affine_coupling_args_A),
            #     flow.nn.ActNorm(cm),
            #     flow.nn.PiecewiseConformalConv2D(cm),
            #     flow.nn.AffineCoupling(**affine_coupling_args_B),
            # ]),
            # rqs coupling flow
            "u_flow": flow.Compose([
                flow.nn.ActNorm(cm),
                flow.nn.PiecewiseConformalConv2D(cm),
                flow.nn.RQSCoupling(**rqs_coupling_args_A),
                flow.nn.ActNorm(cm),
                flow.nn.PiecewiseConformalConv2D(cm),
                flow.nn.RQSCoupling(**rqs_coupling_args_B),
                flow.nn.ActNorm(cm),
                flow.nn.PiecewiseConformalConv2D(cm),
                flow.nn.RQSCoupling(**rqs_coupling_args_A),
                flow.nn.ActNorm(cm),
                flow.nn.PiecewiseConformalConv2D(cm),
                flow.nn.RQSCoupling(**rqs_coupling_args_B),
            ]),
            "dist": flow.distributions.StandardNormal(cm, h4, w4),
        }
    )

    # set up optimizer
    optim_config = {"params": model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.AdamW(**optim_config)

    return model, optim


def train_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    **kwargs,
) -> dict:
    
    assert config.train_loader is not None

    # initialize loop
    model.train()
    data_loader = config.train_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0

    # training
    set_requires_grad(model, True)
    with torch.enable_grad():
        iterable = tqdm(data_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        u_flow: flow.FlowTransform = model["u_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        z_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward/backward pass
            # x |-> u
            u, logabsdet_xu = x_flow(x)
            # m <-| u
            m, _ = x_flow.inverse(u)
            # u <-> z
            z, logabsdet_uz = u_flow(u)

            # TODO fix this
            y_x = y

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_x.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, m, y_x)

            # log likelihood
            log_px = dist.log_prob(z) + logabsdet_xu + logabsdet_uz

            # calculate loss
            reconstruction_loss = torch.nn.functional.mse_loss(m, x)
            nll_loss = -torch.mean(log_px)
            nll_loss = torch.clamp(nll_loss, min=0)
            w = 1e-1
            minibatch_loss = w * nll_loss + (1 - w) * reconstruction_loss

            # backward pass
            optim.zero_grad()
            minibatch_loss.backward()
            optim.step()

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-training
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: [{acc_score[0]:.4f}, {acc_score[1]:.4f}, {acc_score[2]:.4f}]")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "z_pred": np.array(z_pred),
        "samples": samples,
    }


def test_model(
    model: flow.FlowTransform,
    epoch: int,
    config: Config,
    **kwargs,
) -> dict:
    
    assert config.test_loader is not None

    # initialize loop
    model.eval()
    data_loader = config.test_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0

    # testing
    set_requires_grad(model, False)
    with torch.no_grad():
        iterable = tqdm(data_loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        u_flow: flow.FlowTransform = model["u_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        z_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward/backward pass
            # x |-> u
            u, logabsdet_xu = x_flow(x)
            # m <-| u
            m, _ = x_flow.inverse(u)
            # u <-> z
            z, logabsdet_uz = u_flow(u)

            # TODO fix this
            y_x = y

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_x.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, m, y_x)

            # log likelihood
            log_px = dist.log_prob(z) + logabsdet_xu + logabsdet_uz

            # calculate loss
            reconstruction_loss = torch.nn.functional.mse_loss(m, x)
            nll_loss = -torch.mean(log_px)
            nll_loss = torch.clamp(nll_loss, min=0)
            w = 1e-1
            minibatch_loss = w * nll_loss + (1 - w) * reconstruction_loss

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-testing
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: [{acc_score[0]:.4f}, {acc_score[1]:.4f}, {acc_score[2]:.4f}]")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "z_pred": np.array(z_pred),
        "samples": samples,
    }
