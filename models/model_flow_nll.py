from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

from config import Config

from . import flow
from .common import compute_flow_shapes, gather_samples, edl_loss, edl_probs
from .flow.util import decode_mask
from .resnet.residual_block import get_encoder


def compute_sizes(config: Config) -> dict:
    assert config.image_chw
    C, H, W = config.image_chw
    (_, _, _, _, c2, cm, h2, w2, _, _) = compute_flow_shapes(config)

    return {
        "x": (C, H, W),
        "uv": (c2, h2, w2),
        "u": (cm, h2, w2),
        "v": (c2 - cm, h2, w2),
        "z": (cm * h2 * w2),
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

    # Base Distribution
    dist_z = flow.distributions.StandardNormal(k=cm * h2 * w2, mu=0.0, std=1.0)
    dist_v = flow.distributions.StandardNormal(k=(c2 - cm) * h2 * w2, mu=0.0, std=0.01)

    # MNIST: (1, 32, 32) <-> (16, 8, 8) <-> (64, 4, 4)
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

    # MNIST: (cm, 4, 4) <-> (cm * 4 * 4)
    flow_u = flow.Compose(
        [
            #
            flow.nn.ActNorm(cm),
            #
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            flow.nn.RQSCoupling(**rqs_coupling_args_uB),
            flow.nn.RQSCoupling(**rqs_coupling_args_uA),
            #
            flow.nn.Flatten(input_shape=(cm, h2, w2)),
            flow.nn.ActNorm(cm * h2 * w2),
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
            "dist_v": dist_v,
            "flow_x": flow_x,
            "flow_u": flow_u,
            "classifier": classifier,
        }
    )

    # set up optimizer
    optim_g = torch.optim.AdamW(
        params=[*flow_x.parameters(), *flow_u.parameters()],
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
    torchinfo.summary(model["flow_u"], input_size=(B, *sizes["u"]), depth=5)
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
        x_ucty = []
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

            # x -> (u x v) -> u -> z
            uv_x, _ = flow_x(x, forward=True)
            u_x, v_x = flow.nn.partition(uv_x, sizes["u"][0])
            z_x, _ = flow_u(u_x, forward=True)
            uX = (v_x).pow(2).flatten(start_dim=1).sum(dim=-1)

            # u -> (u x 0) -> x~
            u_z = u_x
            v_z = v_x.new_zeros(v_x.size())
            uv_z = flow.nn.join(u_z, v_z)
            x_z, _ = flow_x(uv_z, forward=False)

            # classifier
            y_x = classifier(x)
            pY, uY = edl_probs(
                y_x.detach()
            )  # pY = belief, uY = disbelief ; sum(pY) + uY = 1

            # manifold losses
            L_z = -dist_z.log_prob(z_x)
            L_v = -dist_v.log_prob(v_x)
            L_m = (x - x_z).pow(2).flatten(start_dim=1).sum(dim=-1)

            # classification losses
            # L_y_x = F.cross_entropy(y_x, y) - used EDL alternative
            L_y_x = edl_loss(y_x, y, epoch)

            # accumulate predictions
            u_pred.extend(u_x.detach().flatten(start_dim=1).cpu().numpy())
            v_pred.extend(v_x.detach().flatten(start_dim=1).cpu().numpy())
            y_true.extend(y.argmax(-1).cpu().numpy())
            y_pred.extend(pY.detach().cpu().numpy())
            y_ucty.extend(uY.detach().cpu().numpy())
            x_ucty.extend(uX.detach().cpu().numpy())
            z_pred.extend(z_x.detach().flatten(start_dim=1).cpu().numpy())
            z_nll.extend(L_z.detach().cpu().numpy())
            gather_samples(samples, x, y, x_z, y_x)

            # compute minibatch loss
            ß = 1e-3
            # L_minibatch = L_z.mean() + (ß * L_v.mean()) + L_m.mean() + L_y_x.mean() # Σ_i[Σ_n(L_{n,i})/N]
            L_minibatch = (L_z + (ß * L_v) + L_m + L_y_x).mean()  # Σ_n[Σ_i(L_{n,i})]/N

            # backward pass (if optimizer is given)
            if optim:
                (optim_g, optim_d) = optim
                optim_g.zero_grad()
                optim_d.zero_grad()
                L_minibatch.backward()
                optim_d.step()
                optim_g.step()

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
        "x_ucty": np.array(x_ucty),
        "u_pred": np.array(u_pred),
        "v_pred": np.array(v_pred),
        "z_pred": np.array(z_pred),
        "z_nll": np.array(z_nll),
        "samples": samples,
    }
