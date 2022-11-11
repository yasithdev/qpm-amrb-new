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

    model = torch.nn.ModuleDict(
        {
            "x_flow": flow.CompositeFlowTransform(
                transforms=[
                    # MNIST: (1, 28, 28) -> (16, 7, 7)
                    flow.ops.ConformalConv2D_KxK(c0, k0), #*
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(c1 // 2, 3)
                    ),
                    flow.ops.ConformalActNorm(c1, cm),
                    flow.ops.ConformalConv2D_1x1(c1),
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(c1 // 2, 3)
                    ),
                    flow.ops.ConformalActNorm(c1, cm),
                    flow.ops.ConformalConv2D_1x1(c1), #*
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(c1 // 2, 3)
                    ),
                    flow.ops.ConformalActNorm(c1, cm),
                    flow.ops.ConformalConv2D_1x1(c1),
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(c1 // 2, 3)
                    ),
                    flow.ops.ConformalActNorm(c1, cm),
                ]
            ),
            "m_flow": flow.CompositeFlowTransform(
                # MNIST: (cm, 7, 7) -> (cm * 7 * 7)
                transforms=[
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(cm // 2, 3)
                    ),
                    flow.ops.ActNorm(cm),
                    flow.ops.Conv2D_1x1(cm),
                    flow.ops.ActNorm(cm),
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(cm // 2, 3)
                    ),
                    flow.ops.ActNorm(cm),
                    flow.ops.Conv2D_1x1(cm),
                    flow.ops.ActNorm(cm),
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(cm // 2, 3)
                    ),
                    flow.ops.ActNorm(cm),
                    flow.ops.Conv2D_1x1(cm),
                    flow.ops.ActNorm(cm),
                    flow.ops.AffineCoupling(
                        flow.ops.CouplingNetwork(cm // 2, 3)
                    ),
                    flow.ops.ActNorm(cm),
                    flow.ops.ConformalConv2D_KxK(cm, (h1,w1)),
                    flow.ops.ActNorm(cm * h1 * w1),
                ]
            )
            # mnist - (m, 7, 7)
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
        x_flow: flow.FlowTransform = model["x_flow"] # type: ignore
        m_flow: flow.FlowTransform = model["m_flow"] # type: ignore
        
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
            # project m -> u'
            u_r = flow.util.pad(m, cu - cm)
            # flow u' -> x'
            x_r, ux_logabsdet = x_flow.inverse(u_r)
            # flow m -> z
            z, mz_logabsdet  = m_flow.forward(m)

            # Compute log likelihood
            # log_pu = m_flow.log_prob(m)
            # log_likelihood = torch.mean(log_pu - log_conf_det)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

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
        x_flow: flow.FlowTransform = model["x_flow"] # type: ignore
        m_flow: flow.FlowTransform = model["m_flow"] # type: ignore

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
            # project m -> u'
            u_r = flow.util.pad(m, cu - cm)
            # flow u' -> x'
            x_r, ux_logabsdet = x_flow.inverse(u_r)
            # flow m -> z
            z, mz_logabsdet  = m_flow.forward(m)

            # accumulate predictions TODO fix this
            y_true.extend(torch.argmax(y, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(y, dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_r, y)

            # calculate loss
            minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

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
