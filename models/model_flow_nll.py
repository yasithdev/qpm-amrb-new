from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import edl_loss, edl_probs, gather_samples
from .flow.util import decode_mask


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.image_chw
    assert config.dataset_info

    C, H, W = config.image_chw
    CHW = C * H * W
    D = config.manifold_d
    num_bins = 10

    # ambient (x) flow configuration
    k0, k1 = 4, 2
    c1, h1, w1 = C * k0 * k0, H // k0, W // k0
    c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

    # categorical configuration
    num_labels = config.dataset_info["num_train_labels"]

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

    # non-spatial flow
    u_mask = torch.zeros(D).bool()
    u_mask[::2] = True
    uI, uT = decode_mask(u_mask)
    rqs_coupling_args_uA = {
        "i_channels": uI,
        "t_channels": uT,
        "num_bins": num_bins,
        "spatial": False,
    }
    rqs_coupling_args_uB = {
        "i_channels": uT,
        "t_channels": uI,
        "num_bins": num_bins,
        "spatial": False,
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
            #
            flow.nn.Flatten(input_shape=(c2, h2, w2)),
        ]
    )

    # (B,D) <-> (B,D)
    flow_u = flow.Compose(
        [
            #
            flow.nn.ActNorm(D),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            flow.nn.RQSCoupling(**rqs_coupling_args_uB),
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            #
            flow.nn.ActNorm(D),
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
    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim_lr,
    )

    return model, (optim,)


def describe_model(
    model: torch.nn.ModuleDict,
    config: Config,
) -> None:

    assert config.image_chw

    B = config.batch_size
    D = config.manifold_d
    (C, H, W) = config.image_chw

    torchinfo.summary(model["flow_x"], input_size=(B, C, H, W), depth=5)
    torchinfo.summary(model["flow_u"], input_size=(B, D), depth=5)
    torchinfo.summary(model["classifier"], input_size=(B, C, H, W), depth=5)


def step_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
    **kwargs,
) -> dict:

    # pre-step
    D = config.manifold_d
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
            u_x, v_x = flow.nn.partition(uv_x, D)
            z_x, _ = flow_u(u_x, forward=True)

            # u -> (u x 0) -> x~
            u_z = u_x
            v_z = v_x.new_zeros(v_x.size())
            uv_z = flow.nn.join(u_z, v_z)
            x_z, _ = flow_x(uv_z, forward=False)

            # classifier
            y_z = classifier(z_x)
            pY, uY = edl_probs(y_z.detach())

            # manifold losses
            L_z = -dist_z.log_prob(z_x)
            L_v = -dist_v.log_prob(v_x)
            L_m = (x - x_z).pow(2).flatten(1).sum(-1)

            # classification losses
            # L_y_x = F.cross_entropy(y_x, y) - used EDL alternative
            L_y_x = edl_loss(y_z, y, epoch)

            # accumulate predictions
            u_norm.extend(u_x.detach().flatten(1).norm(2, -1).cpu().numpy())
            v_norm.extend(v_x.detach().flatten(1).norm(2, -1).cpu().numpy())
            z_norm.extend(z_x.detach().flatten(1).norm(2, -1).cpu().numpy())

            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(pY.detach().cpu().numpy())
            y_ucty.extend(uY.detach().cpu().numpy())
            z_nll.extend(L_z.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_z)

            # compute minibatch loss
            ß = 1e-3
            # L_minibatch = L_z.mean() + (ß * L_v.mean()) + L_m.mean() + L_y_x.mean() # Σ_i[Σ_n(L_{n,i})/N]
            L_minibatch = (L_z + (ß * L_v) + L_m + L_y_x).mean()  # Σ_n[Σ_i(L_{n,i})]/N

            # backward pass (if optimizer is given)
            if optim:
                (optim_gd,) = optim
                optim_gd.zero_grad()
                L_minibatch.backward()
                optim_gd.step()

            # accumulate sum loss
            sum_loss += L_minibatch.item() * config.batch_size

            # logging
            log_stats = {
                "Loss(mb)": f"{L_minibatch:.4f}",
                "Loss(z)": f"{L_z.mean():.4f}",
                "Loss(v)": f"{L_v.mean():.4f}",
                "Loss(m)": f"{L_m.mean():.4f}",
                "Loss(y)": f"{L_y_x.mean():.4f}",
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
        "z_nll": np.array(z_nll),
        "u_norm": np.array(u_norm),
        "v_norm": np.array(v_norm),
        "z_norm": np.array(z_norm),
    }
