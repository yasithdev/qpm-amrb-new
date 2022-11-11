from typing import Tuple

import torch
from tqdm import tqdm

from config import Config

from . import flow
from .common import gather_samples, gen_epoch_acc, set_requires_grad


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, torch.optim.Optimizer]:

    # network configuration
    k0 = 4
    c0, h0, w0 = config.image_chw
    c1, h1, w1 = c0 * k0 * k0, h0 // k0, w0 // k0
    cm = config.manifold_c
    cms = cm // 2
    blocks = 3

    model = torch.nn.ModuleDict(
        {
            # MNIST: (1, 28, 28) -> (16, 7, 7)
            "x_flow": flow.Compose(
                flow.nn.ConformalConv2D_KxK(c0, k0),
                flow.nn.ConformalActNorm(c1),
                flow.nn.ConformalConv2D_1x1(c1),
                flow.nn.ConformalActNorm(c1),
                flow.nn.ConformalConv2D_1x1(c1),
                flow.nn.ConformalActNorm(c1),
                flow.nn.ConformalConv2D_1x1(c1),
                flow.nn.ConformalActNorm(c1),
            ),
            # MNIST: (cm, 7, 7) -> (cm, 7, 7)
            "m_flow": flow.Compose(
                flow.nn.AffineCoupling(flow.nn.CouplingNetwork(cms, blocks)),
                flow.nn.Conv2D_1x1(cm),
                flow.nn.ActNorm(cm),
                flow.nn.AffineCoupling(flow.nn.CouplingNetwork(cms, blocks)),
                flow.nn.Conv2D_1x1(cm),
                flow.nn.ActNorm(cm),
                flow.nn.AffineCoupling(flow.nn.CouplingNetwork(cms, blocks)),
                flow.nn.Conv2D_1x1(cm),
                flow.nn.ActNorm(cm),
                flow.nn.AffineCoupling(flow.nn.CouplingNetwork(cms, blocks)),
                flow.nn.Conv2D_1x1(cm),
                flow.nn.ActNorm(cm),
            ),
            "dist": flow.distributions.StandardNormal(cm, h1, w1)
        }
    )

    # set up optimizer
    optim_config = {"params": model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.Adam(**optim_config)

    return model, optim


def train_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: torch.optim.Optimizer,
    **kwargs,
) -> dict:

    # initialize loop
    model.train()
    data_loader = config.train_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0
    cm = config.manifold_c

    # training
    set_requires_grad(model, True)
    with torch.enable_grad():

        iterable = tqdm(data_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        m_flow: flow.FlowTransform = model["m_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"] # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            # flow x -> u
            u, xu_logabsdet = x_flow.forward(x)
            cu = u.size(1)
            # project u -> m
            m = flow.util.proj(u, manifold_dims=cm)
            # flow m -> z
            z, mz_logabsdet = m_flow.forward(m)

            # inverse pass
            # project m -> u'
            u_r = flow.util.pad(m, cu - cm)
            # flow u' -> x'
            x_r, ux_logabsdet = x_flow.inverse(u_r)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # log likelihood
            log_px = dist.log_prob(z) + mz_logabsdet + .5 * xu_logabsdet

            # calculate loss
            reconstruction_loss = torch.nn.functional.mse_loss(x_r, x)
            nll_loss = -torch.mean(log_px)
            w = 1e-2
            minibatch_loss = w * torch.clamp(nll_loss, min=0) + (1-w) * reconstruction_loss

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

    tqdm.write(f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": samples,
    }


def test_model(
    model: flow.FlowTransform,
    epoch: int,
    config: Config,
    **kwargs,
) -> dict:

    # initialize loop
    model.eval()
    data_loader = config.test_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0
    cm = config.manifold_c

    # testing
    set_requires_grad(model, False)
    with torch.no_grad():
        iterable = tqdm(data_loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        m_flow: flow.FlowTransform = model["m_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"] # type: ignore

        x: torch.Tensor
        y: torch.Tensor
        y_true = []
        y_pred = []
        samples = []

        for x, y in iterable:

            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)

            # forward pass
            # flow x -> u
            u, xu_logabsdet = x_flow.forward(x)
            cu = u.size(1)
            # project u -> m
            m = flow.util.proj(u, manifold_dims=cm)
            # flow m -> z
            z, mz_logabsdet = m_flow.forward(m)

            # inverse pass
            # project m -> u'
            u_r = flow.util.pad(m, cu - cm)
            # flow u' -> x'
            x_r, ux_logabsdet = x_flow.inverse(u_r)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # log likelihood
            log_px = dist.log_prob(z) + mz_logabsdet - .5 * ux_logabsdet

            # calculate loss
            reconstruction_loss = torch.nn.functional.mse_loss(x_r, x)
            nll_loss = -torch.mean(log_px)
            w = 1e-3
            minibatch_loss = w * nll_loss + (1-w) * reconstruction_loss

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-testing
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: {acc_score:.4f}")

    return {
        "loss": avg_loss,
        "acc": acc_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "samples": samples,
    }
