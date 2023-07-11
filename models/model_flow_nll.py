from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import edl_loss, edl_probs, gather_samples
from .flow.util import decode_mask

cm, k0, k1 = 0, 4, 2


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:
    assert config.image_chw is not None

    C, H, W = config.image_chw
    CHW = C * H * W
    D = config.manifold_d
    num_bins = 10

    # ambient (x) flow configuration
    c1, h1, w1 = C * k0 * k0, H // k0, W // k0
    c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

    # adjust manifold_d to closest possible value
    global cm
    cm = D // (h2 * w2)
    D = cm * h2 * w2
    config.manifold_d = D

    # categorical configuration
    ind_targets = config.get_ind_labels()
    num_labels = len(ind_targets)

    # spatial flow
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

    # reduced-channel flow
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

    # Base Distributions
    dist_z = flow.distributions.StandardNormal(k=D, mu=0.0, std=1.0)
    dist_v = flow.distributions.StandardNormal(k=CHW - D, mu=0.0, std=0.01)

    # (B,C,H,W)->(B,16*C,H/4,H/4)->flow->(B,16*C,H/4,H/4)->(B,64*C,H/8,W/8)->flow->(B,64*C,H/8,W/8)->(B,C*H*W)
    flow_x = flow.Compose(
        [
            #
            flow.nn.Squeeze(factor=k0),
            flow.nn.ActNorm(c1),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
            flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
            flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
            #
            flow.nn.Squeeze(factor=k1),
            flow.nn.ActNorm(c2),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
            flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
            flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
        ]
    )

    # (B,D) <-> (B,D)
    flow_u = flow.Compose(
        [
            #
            flow.nn.ActNorm(cm),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            flow.nn.RQSCoupling(**rqs_coupling_args_uB),
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            #
            flow.nn.ActNorm(cm),
            #
            flow.nn.Flatten((cm, h2, w2)),
        ]
    )

    # (B,D) -> (B,K)
    classifier = torch.nn.Sequential(
        torch.nn.Linear(D, D),
        torch.nn.Tanh(),
        torch.nn.Linear(D, num_labels),
    )

    model = torch.nn.ModuleDict(
        {
            "dist_z": dist_z,
            "dist_v": dist_v,
            "flow_x": flow_x,
            "flow_u": flow_u,
            "classifier": classifier,
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

    B = config.batch_size
    D = config.manifold_d
    f = k0 * k1
    (C, H, W) = config.image_chw
    global cm
    torchinfo.summary(model["flow_x"], input_size=(B, C, H, W), depth=5)
    torchinfo.summary(model["flow_u"], input_size=(B, cm, H // f, W // f), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, D), depth=5)


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
        dist_z: flow.distributions.Distribution = model["dist_z"]  # type: ignore
        dist_v: flow.distributions.Distribution = model["dist_v"]  # type: ignore
        flow_x: flow.Compose = model["flow_x"]  # type: ignore
        flow_u: flow.Compose = model["flow_u"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore

        x: torch.Tensor
        y: torch.Tensor

        uv_x: torch.Tensor
        u_x: torch.Tensor
        v_x: torch.Tensor
        z_x: torch.Tensor

        u_z: torch.Tensor
        v_z: torch.Tensor
        x_z: torch.Tensor

        y_true = []
        y_pred = []
        y_ucty = []
        u_norm = []
        v_norm = []
        z_norm = []
        z_nll = []
        samples = []

        for x, y in iterable:
            # cast x and y to float
            x = x.float().to(config.device)
            y = y.float().to(config.device)
            B = x.size(0)

            # x -> (u x v) -> u -> z
            uv_x, _ = flow_x(x, forward=True)
            u_x, v_x = flow.nn.partition(uv_x, cm)
            z_x, _ = flow_u(u_x, forward=True)

            # u -> (u x 0) -> x~
            u_z = u_x
            v_z = v_x.new_zeros(v_x.size())
            uv_z = flow.nn.join(u_z, v_z)
            x_z, _ = flow_x(uv_z, forward=False)

            # classifier
            y_z = classifier(u_x.flatten(1))
            pY, uY = edl_probs(y_z.detach())

            # manifold losses
            L_z = -dist_z.log_prob(z_x.flatten(1))
            L_v = -dist_v.log_prob(v_x.flatten(1))
            L_x = (x - x_z).flatten(1).pow(2).sum(-1)

            # classification losses
            # L_y_z = F.cross_entropy(y_x, y) - used EDL alternative
            L_y_z = edl_loss(y_z, y, epoch)

            # accumulate predictions
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(pY.detach().cpu().numpy())
            y_ucty.extend(uY.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # accumulate extra predictions
            u_norm.extend(u_x.detach().flatten(1).norm(2, -1).cpu().numpy())
            v_norm.extend(v_x.detach().flatten(1).norm(2, -1).cpu().numpy())
            z_norm.extend(z_x.detach().flatten(1).norm(2, -1).cpu().numpy())
            z_nll.extend(L_z.detach().cpu().numpy())

            # compute minibatch loss
            ß = 1e-3
            # L_minibatch = L_z.mean() + (ß * L_v.mean()) + L_m.mean() + L_y_z.mean() # Σ_i[Σ_n(L_{n,i})/N]
            L_minibatch = (L_z + (ß * L_v) + L_x + L_y_z).mean()  # Σ_n[Σ_i(L_{n,i})]/N

            # backward pass (if optimizer is given)
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
                "Loss(x)": f"{L_x.mean():.4f}",
                "Loss(y)": f"{L_y_z.mean():.4f}",
                "Loss(v)": f"{L_v.mean():.4f}",
                "Loss(z)": f"{L_z.mean():.4f}",
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
        "u_norm": np.array(u_norm),
        "v_norm": np.array(v_norm),
        "z_norm": np.array(z_norm),
        "z_nll": np.array(z_nll),
    }
