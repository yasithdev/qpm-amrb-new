from abc import abstractmethod
from typing import Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nets as nets
from . import FlowTransform


class CouplingTransform(FlowTransform):
    """
    A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). Images are split on the channel dimension, using the provided 1D mask.

    """

    def __init__(
        self,
        i_channels: torch.Tensor,
        t_channels: torch.Tensor,
        coupling_net: nn.Module,
    ) -> None:

        super().__init__()

        self.i_channels = i_channels
        self.t_channels = t_channels
        self.coupling_net = coupling_net

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4], "Inputs must be a 2D or a 4D tensor."

        xI = x[:, self.i_channels, ...]
        xT = x[:, self.t_channels, ...]

        params = self.coupling_net(xI, c)
        zI = xI
        zT, logabsdet = self.coupling_transform(xT, params, inverse=False)

        z = torch.empty_like(x)
        z[:, self.i_channels, ...] = zI
        z[:, self.t_channels, ...] = zT

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4], "Inputs must be a 2D or a 4D tensor."

        zI = z[:, self.i_channels, ...]
        zT = z[:, self.t_channels, ...]

        xI = zI
        params = self.coupling_net(xI, c)
        xT, logabsdet = self.coupling_transform(zT, params, inverse=True)

        x = torch.empty_like(z)
        x[:, self.i_channels, ...] = xI
        x[:, self.t_channels, ...] = xT

        return x, logabsdet

    @abstractmethod
    def coupling_transform(
        self,
        data: torch.Tensor,
        params: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coupling Transform Function"""
        raise NotImplementedError()


class AffineCoupling(CouplingTransform):
    """
    (p1) ---------------(p1)
             ↓     ↓
            (s)   (t)
             ↓     ↓
    (p2) ---[x]---[+]---(p2)
    """

    def __init__(
        self,
        i_channels: torch.Tensor,
        t_channels: torch.Tensor,
        num_layers: int,
    ) -> None:

        nI = len(i_channels)
        nT = len(t_channels)
        assert nI == nT, f"channel masks have mismatching sizes! ({nI} and {nT})"

        coupling_net = nets.Conv2DResNet(
            in_channels=nI,
            out_channels=nT * 2,
            hidden_channels=nI,
        )

        super().__init__(
            i_channels=i_channels,
            t_channels=t_channels,
            coupling_net=coupling_net,
        )

        self.nI = nI
        self.nT = nT

    def coupling_transform(
        self,
        data: torch.Tensor,
        params: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s, t = torch.chunk(params, 2, dim=1)
        sign = -1 if inverse else +1

        if inverse:
            output = (data - t) / s.exp()
        else:
            output = s.exp() * data + t

        logabsdet = sign * einops.reduce(s, "B ... -> B", reduction="sum")
        return output, logabsdet


class RQSCoupling(CouplingTransform):
    """
    (p1) ---------------(p1)
                ↓
            ([w,h,d])        <- piecewise RQS functions
                ↓
    (p2) ---- f(.) ---- (p2)
    """

    def __init__(
        self,
        i_channels: torch.Tensor,
        t_channels: torch.Tensor,
        num_bins: int = 10,
        bound: float = 10.0,
        min_w: float = 1e-3,
        min_h: float = 1e-3,
        min_d: float = 1e-3,
    ) -> None:

        nI = len(i_channels)
        nT = len(t_channels)

        assert nI == nT, f"channel masks have mismatching sizes! ({nI} and {nT})"
        assert min_w * num_bins <= 1.0, "min_w too large for the number of bins"
        assert min_h * num_bins <= 1.0, "min_h too large for the number of bins"

        nW = num_bins
        nH = num_bins
        nD = num_bins - 1

        coupling_net = nets.Conv2DResNet(
            in_channels=nI,
            out_channels=nT * (nW + nH + nD),
            hidden_channels=nI,
        )

        super().__init__(
            i_channels=i_channels,
            t_channels=t_channels,
            coupling_net=coupling_net,
        )

        self.nI = nI
        self.nT = nT

        self.nW = nW
        self.nH = nH
        self.nD = nD

        self.num_bins = num_bins
        self.lo = -bound
        self.hi = +bound
        self.min_w = min_w
        self.min_h = min_h
        self.min_d = min_d

    def coupling_transform(
        self,
        data: torch.Tensor,
        params: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # rearrange params
        params = einops.rearrange(
            params,
            "B (C P) H W -> B C H W P",
            C=data.size(1),
            P=self.nH + self.nW + self.nD,
        )
        w, h, d = einops.unpack(
            tensor=params,
            packed_shapes=[(self.nW,), (self.nH,), (self.nD,)],
            pattern="B C H W *",
        )

        # constant-pad d for outer data
        # const = np.log(np.exp(1 - self.min_d) - 1)
        const = 1.0
        d = F.pad(d, pad=(1, 1), value=const)

        assert w.shape[-1] == self.num_bins
        assert h.shape[-1] == self.num_bins
        assert d.shape[-1] == self.num_bins + 1

        output = torch.empty_like(data)
        logabsdet = torch.empty_like(data)

        idx_outer = (data <= self.lo) | (data >= self.hi)
        idx_inner = ~idx_outer

        # outer data - identity transform
        output[idx_outer], logabsdet[idx_outer] = data[idx_outer], 0.0

        # inner data - rqs transform
        kwargs = {
            "data": data[idx_inner],
            "h": h[idx_inner, :],
            "w": w[idx_inner, :],
            "d": d[idx_inner, :],
            "inverse": inverse,
        }
        output[idx_inner], logabsdet[idx_inner] = self.spline(**kwargs)
        logabsdet = einops.reduce(logabsdet, "b ... -> b", reduction="sum")

        return output, logabsdet

    def spline(
        self,
        data: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        assert w.shape[-1] == self.num_bins
        assert h.shape[-1] == self.num_bins
        assert d.shape[-1] == self.num_bins + 1

        left = bottom = self.lo
        right = top = self.hi

        # scale down the variance
        h /= np.sqrt(self.nI)
        w /= np.sqrt(self.nI)

        heights = self.min_h + (1 - self.min_h * self.num_bins) * F.softmax(h, dim=-1)
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        # rounding to prevent errors
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        widths = self.min_w + (1 - self.min_w * self.num_bins) * F.softmax(w, dim=-1)
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        # rounding to prevent errors
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivs = self.min_d + F.softplus(d)

        if inverse:
            bin_idx = self.searchsorted(cumheights, data)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, data)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivs = derivs.gather(-1, bin_idx)[..., 0]
        input_derivs_plus_one = derivs[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (data - input_cumheights) * (
                input_derivs + input_derivs_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivs)
            b = input_heights * input_derivs - (data - input_cumheights) * (
                input_derivs + input_derivs_plus_one - 2 * input_delta
            )
            c = -input_delta * (data - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            # root = (- b + torch.sqrt(discriminant)) / (2 * a)
            output = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivs + input_derivs_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivs_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivs * (1 - root).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return output, -logabsdet

        else:
            theta = (data - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivs * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivs + input_derivs_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            output = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivs_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivs * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return output, logabsdet

    @staticmethod
    def searchsorted(
        bin_locations: torch.Tensor,
        inputs: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:

        bin_locations[..., -1] += eps
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
