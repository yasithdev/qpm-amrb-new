from typing import Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config

from . import flow
from .common import gather_samples, gen_epoch_acc, set_requires_grad, npad
from .flow.util import decode_mask
from .resnet import ResidualBlock

import numpy as np


class GradientScaler(torch.autograd.Function):
    factor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        factor = GradientScaler.factor
        return factor.view(-1, 1, 1, 1) * grad_output
        # return torch.neg(grad_output)


def get_gradient_ratios(
    lossA: torch.Tensor,
    lossB: torch.Tensor,
    x_f: torch.Tensor,
) -> torch.Tensor:

    grad_lossA_xf = torch.autograd.grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = torch.autograd.grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma


def get_encoder(
    input_chw: Tuple[int, int, int],
    num_features: int,
) -> torch.nn.Module:
    """
    Create a ResNet-Based Encoder

    :param input_chw: (C, H, W)
    :param num_features: number of output channels (D)
    :return: model that transforms (B, C, H, W) -> (B, D, 1, 1)
    """
    (c, h, w) = input_chw
    d = num_features

    k1, s1 = 5, 2
    pw1, ph1 = npad(w, k1, s1), npad(h, k1, s1)

    c1 = 32
    c2 = 48
    c3 = 64

    model = torch.nn.Sequential(
        # (B,C,H,W) -> (B,C,H+ph,W+pw)
        torch.nn.ZeroPad2d((0, pw1, 0, ph1)),
        # (B,C,H+ph,W+pw) -> (B,C1,H/2,W/2)
        torch.nn.Conv2d(
            in_channels=c,
            out_channels=c1,
            kernel_size=k1,
            stride=s1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(c1),
        torch.nn.ReLU(),
        # (B,C1,H/2,W/2) -> (B,C1,H/2,W/2)
        ResidualBlock(
            in_channels=c1,
            hidden_channels=c1,
            out_channels=c1,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B,C1,H/2,W/2) -> (B,C2,H/4,W/4)
        ResidualBlock(
            in_channels=c1,
            hidden_channels=c1,
            out_channels=c2,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B,C2,H/4,W/4) -> (B,C2,H/4,W/4)
        ResidualBlock(
            in_channels=c2,
            hidden_channels=c2,
            out_channels=c2,
            stride=1,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B,C2,H/4,W/4) -> (B,C3,H/8,W/8)
        ResidualBlock(
            in_channels=c2,
            hidden_channels=c2,
            out_channels=c3,
            stride=2,
            conv=torch.nn.Conv2d,
            norm=torch.nn.BatchNorm2d,
        ),
        # (B,C3,H/8,W/8) -> (B,D,1,1)
        torch.nn.Conv2d(
            in_channels=c3,
            out_channels=d,
            kernel_size=(h // 8, w // 8),
            stride=1,
        ),
        torch.nn.BatchNorm2d(d),
        torch.nn.Flatten(),
    )

    return model


def compute_shape_params(
    config: Config,
) -> Tuple[int, ...]:

    assert config.image_chw
    assert config.dataset_info

    k0, k1 = 4, 2

    # ambient (x) flow configuration
    c0, h0, w0 = config.image_chw
    c1, h1, w1 = c0 * k0 * k0, h0 // k0, w0 // k0
    c2, h2, w2 = c1 * k1 * k1, h1 // k1, w1 // k1

    # manifold (m) flow configuration
    cm = config.manifold_d // h2 // w2
    num_bins = 10

    # categorical configuration
    num_labels = config.dataset_info["num_train_labels"]
    return (k0, k1, c0, c1, c2, cm, h2, w2, num_bins, num_labels)


def load_model_and_optimizer(
    config: Config,
) -> Tuple[torch.nn.ModuleDict, Tuple[torch.optim.Optimizer, ...]]:

    assert config.dataset_info is not None
    assert config.image_chw is not None

    (k0, k1, c0, c1, c2, cm, h2, w2, num_bins, num_labels) = compute_shape_params(
        config
    )

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
            # MNIST: (1, 32, 32) -> (16, 8, 8)  -> (64, 4, 4)
            "x_flow": flow.Compose(
                [
                    flow.nn.ActNorm(c0),
                    flow.nn.Squeeze(factor=k0),
                    flow.nn.PiecewiseConformalConv2D(c1),
                    flow.nn.ActNorm(c1),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1A),
                    flow.nn.ActNorm(c1),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x1B),
                    flow.nn.ActNorm(c1),
                    flow.nn.Squeeze(factor=k1),
                    flow.nn.PiecewiseConformalConv2D(c2),
                    flow.nn.ActNorm(c2),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2A),
                    flow.nn.ActNorm(c2),
                    flow.nn.RQSCoupling(**rqs_coupling_args_x2B),
                    flow.nn.ActNorm(c2),
                    flow.nn.PiecewiseConformalConv2D(c2),
                    flow.nn.ActNorm(c2),
                ]
            ),
            # MNIST: (cm, 4, 4) -> (cm * 4 * 4)
            "u_flow": flow.Compose(
                [
                    flow.nn.ActNorm(cm),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uA),
                    flow.nn.ActNorm(cm),
                    flow.nn.RQSCoupling(**rqs_coupling_args_uB),
                    flow.nn.Flatten(input_shape=(cm, h2, w2)),
                ]
            ),
            "dist": flow.distributions.StandardNormal(cm * h2 * w2 - num_labels),
            "classifier": torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=cm * h2 * w2, out_features=128),
                torch.nn.SELU(),
                torch.nn.Linear(in_features=128, out_features=64),
                torch.nn.SELU(),
                torch.nn.Linear(in_features=64, out_features=num_labels),
            ),
            "critic": torch.nn.Sequential(
                get_encoder(
                    input_chw=config.image_chw,
                    num_features=1,
                ),
                torch.nn.Sigmoid(),
            ),
        }
    )

    # set up optimizer
    optim_config = {"lr": config.optim_lr}
    optim_g = torch.optim.AdamW(
        params=list(model["x_flow"].parameters())
        + list(model["u_flow"].parameters())
        + list(model["classifier"].parameters()),
        **optim_config,
    )
    optim_d = torch.optim.AdamW(
        params=model["critic"].parameters(),
        **optim_config,
    )

    return model, (optim_g, optim_d)


def train_model(
    model: torch.nn.ModuleDict,
    epoch: int,
    config: Config,
    optim: Tuple[torch.optim.Optimizer, ...],
    **kwargs,
) -> dict:

    assert config.train_loader is not None

    # initialize loop
    model.train()
    data_loader = config.train_loader
    size = len(data_loader.dataset)  # type: ignore
    sum_loss = 0
    cm = compute_shape_params(config)[5]

    # training
    set_requires_grad(model, True)
    with torch.enable_grad():
        iterable = tqdm(data_loader, desc=f"[TRN] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        u_flow: flow.FlowTransform = model["u_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore
        critic: torch.nn.Module = model["critic"]  # type: ignore
        scaler = GradientScaler.apply
        cT = torch.ones(config.batch_size, 1).to(config.device)
        cF = torch.zeros(config.batch_size, 1).to(config.device)

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

            u: torch.Tensor
            v: torch.Tensor
            m: torch.Tensor
            z: torch.Tensor
            y_x: torch.Tensor
            logabsdet_xu: torch.Tensor
            logabsdet_uz: torch.Tensor
            logabsdet_zu: torch.Tensor
            logabsdet_ux: torch.Tensor

            # (a) classification pass
            uv_x, logabsdet_xu = x_flow(x, forward=True)
            u_x, v_x = flow.nn.partition(uv_x, cm)
            y_x = classifier(u_x)
            y_x = F.gumbel_softmax(y_x)

            # (b) conditional generation pass
            z = dist.sample(B).to(x.device)
            z = torch.concat([z, y_x.detach()], dim=1)
            u_z, logabsdet_zu = u_flow(z, forward=False)
            y_z = classifier(u_z)
            y_z = F.gumbel_softmax(y_z)
            v_z = torch.zeros_like(v_x)
            uv_z = flow.nn.join(u_z, v_z)
            x_z, logabsdet_ux = x_flow(uv_z, forward=False)
            x_z: torch.Tensor = scaler(x_z) # type: ignore

            # (c) discriminator pass
            c_x: torch.Tensor = critic(x)
            c_z: torch.Tensor = critic(x_z)

            # accumulate predictions
            u_pred.extend(u_x.detach().cpu().numpy())
            v_pred.extend(v_x.detach().cpu().numpy())
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_x.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y_x)

            # calculate losses
            # log_px = dist.log_prob(z) - logabsdet_zu - logabsdet_ux
            # z_nll.extend(-log_px.detach().cpu().numpy())
            # mse_loss = F.mse_loss(x_z, x)

            cT_loss = F.binary_cross_entropy_with_logits(c_x, cT[:B])
            cF_loss = F.binary_cross_entropy_with_logits(c_z, cF[:B], reduction="none")

            d_loss = cT_loss + torch.mean(cF_loss)
            g_loss = F.binary_cross_entropy_with_logits(c_z, cT[:B], reduction="none")

            gamma = get_gradient_ratios(g_loss, cF_loss, c_z)
            GradientScaler.factor = gamma

            y_x_loss = F.cross_entropy(y_x, y)
            y_z_loss = F.cross_entropy(y_z, y)

            # v_loss = F.l1_loss(v, torch.zeros_like(v))
            # z_loss = -log_px.mean()
            # α = 0.01
            # β = 0.1
            minibatch_loss = y_x_loss + y_z_loss + d_loss + g_loss.mean()

            # backward pass
            (optim_g, optim_d) = optim

            optim_g.zero_grad()
            optim_d.zero_grad()

            d_loss.backward()

            optim_g.step()
            optim_d.step()

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-training
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(
        f"[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: [{acc_score[0]:.4f}, {acc_score[1]:.4f}, {acc_score[2]:.4f}]"
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
    cm = compute_shape_params(config)[5]

    # testing
    set_requires_grad(model, False)
    with torch.no_grad():
        iterable = tqdm(data_loader, desc=f"[TST] Epoch {epoch}", **config.tqdm_args)
        x_flow: flow.FlowTransform = model["x_flow"]  # type: ignore
        u_flow: flow.FlowTransform = model["u_flow"]  # type: ignore
        dist: flow.distributions.Distribution = model["dist"]  # type: ignore
        classifier: torch.nn.Module = model["classifier"]  # type: ignore
        critic: torch.nn.Module = model["critic"]  # type: ignore
        scaler = GradientScaler.apply
        cT = torch.ones(config.batch_size, 1).to(config.device)
        cF = torch.zeros(config.batch_size, 1).to(config.device)

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

            u: torch.Tensor
            v: torch.Tensor
            m: torch.Tensor
            z: torch.Tensor
            y_x: torch.Tensor
            logabsdet_xu: torch.Tensor
            logabsdet_uz: torch.Tensor
            logabsdet_zu: torch.Tensor
            logabsdet_ux: torch.Tensor

            # (a) classification pass
            uv_x, logabsdet_xu = x_flow(x, forward=True)
            u_x, v_x = flow.nn.partition(uv_x, cm)
            y_x = classifier(u_x)
            y_x = F.gumbel_softmax(y_x)

            # (b) conditional generation pass
            z = dist.sample(B).to(x.device)
            z = torch.concat([z, y_x.detach()], dim=1)
            u_z, logabsdet_zu = u_flow(z, forward=False)
            y_z = classifier(u_z)
            y_z = F.gumbel_softmax(y_z)
            v_z = torch.zeros_like(v_x)
            uv_z = flow.nn.join(u_z, v_z)
            x_z, logabsdet_ux = x_flow(uv_z, forward=False)
            x_z: torch.Tensor = scaler(x_z) # type: ignore

            # (c) discriminator pass
            c_x: torch.Tensor = critic(x)
            c_z: torch.Tensor = critic(x_z)

            # accumulate predictions
            u_pred.extend(u_x.detach().cpu().numpy())
            v_pred.extend(v_x.detach().cpu().numpy())
            y_true.extend(y.detach().argmax(-1).cpu().numpy())
            y_pred.extend(y_x.detach().softmax(-1).cpu().numpy())
            z_pred.extend(z.detach().flatten(start_dim=1).cpu().numpy())
            gather_samples(samples, x, y, x_z, y_x)

            # calculate losses
            # log_px = dist.log_prob(z) - logabsdet_zu - logabsdet_ux
            # z_nll.extend(-log_px.detach().cpu().numpy())
            # mse_loss = F.mse_loss(x_z, x)

            cT_loss = F.binary_cross_entropy_with_logits(c_x, cT[:B])
            cF_loss = F.binary_cross_entropy_with_logits(c_z, cF[:B], reduction="none")

            d_loss = cT_loss + torch.mean(cF_loss)
            g_loss = F.binary_cross_entropy_with_logits(c_z, cT[:B], reduction="none")

            # gamma = get_gradient_ratios(g_loss, cF_loss, c_z)
            # GradientScaler.factor = gamma

            y_x_loss = F.cross_entropy(y_x, y)
            y_z_loss = F.cross_entropy(y_z, y)

            # v_loss = F.l1_loss(v, torch.zeros_like(v))
            # z_loss = -log_px.mean()
            # α = 0.01
            # β = 0.1
            minibatch_loss = y_x_loss + y_z_loss + d_loss + g_loss.mean()

            # accumulate sum loss
            sum_loss += minibatch_loss.item() * config.batch_size

            # logging
            log_stats = {"Loss(mb)": f"{minibatch_loss.item():.4f}"}
            iterable.set_postfix(log_stats)

    # post-testing
    avg_loss = sum_loss / size
    acc_score = gen_epoch_acc(y_pred=y_pred, y_true=y_true)

    tqdm.write(
        f"[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Acc: [{acc_score[0]:.4f}, {acc_score[1]:.4f}, {acc_score[2]:.4f}]"
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
