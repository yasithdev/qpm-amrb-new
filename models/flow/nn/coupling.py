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

        if x.dim() == 2:
            params = self.coupling_net(xI)
        else:
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
        in_h: int,
        in_w: int,
        i_channels: torch.Tensor,
        t_channels: torch.Tensor,
        num_layers: int,
    ) -> None:
        nI = len(i_channels)
        nT = len(t_channels)
        assert nI == nT, f"channel masks have mismatching sizes! ({nI} and {nT})"

        coupling_net = nets.Conv2DResNet(
            in_h=in_h,
            in_w=in_w,
            in_channels=nI,
            out_channels=nT * 2,
            hidden_channels=nI // 2,
            num_groups=nI // 2,
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
        bound: float = 1.0,
        min_w: float = 1e-2,
        min_h: float = 1e-2,
        min_d: float = 1e-2,
        spatial: bool = True,
        in_h: Optional[int] = None,
        in_w: Optional[int] = None,
        use_groups: bool = True,
    ) -> None:
        nI = len(i_channels)
        nT = len(t_channels)

        assert nI == nT, f"channel masks have mismatching sizes! ({nI} and {nT})"
        assert min_w * num_bins <= (bound * 2), "min_w too large for the number of bins"
        assert min_h * num_bins <= (bound * 2), "min_h too large for the number of bins"

        nW = num_bins
        nH = num_bins
        nD = num_bins - 1
        nParams = nT * (nW + nH + nD)

        num_groups = 1
        if use_groups:
            num_groups = nI // 2

        if spatial:
            assert in_h is not None
            assert in_w is not None
            coupling_net = nets.Conv2DResNet(
                in_h=in_h,
                in_w=in_w,
                in_channels=nI,
                out_channels=nParams,
                hidden_channels=nI // 2,
                num_groups=num_groups,
            )
        else:
            assert in_h is None
            assert in_w is None
            coupling_net = nets.Conv2DNonSpatial(
                in_channels=nI,
                out_channels=nParams,
                hidden_channels=nI // 2,
                num_groups=num_groups,
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
        nW, nH, nD = self.nW, self.nH, self.nD

        # rearrange params
        params = einops.rearrange(
            params, "B (C P) ... -> B C ... P", C=data.size(1), P=nW + nH + nD
        )
        w, h, d = params.tensor_split([nW, nW + nH], dim=-1)

        # constant-pad d for outer data
        # const = np.log(np.exp(1 - self.min_d) - 1)
        const = 1.0
        d = F.pad(d, pad=(1, 1), value=const)

        assert w.shape[-1] == self.num_bins
        assert h.shape[-1] == self.num_bins
        assert d.shape[-1] == self.num_bins + 1

        # default - identity transform
        output = data.clone().detach()
        logabsdet = torch.zeros_like(data)

        # inner data - rqs transform
        idx_inner = (data > self.lo) & (data < self.hi)

        if idx_inner.numel() > 0:
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
        left = bottom = self.lo
        right = top = self.hi

        heights = self.min_h + (1 - self.min_h * self.num_bins) * F.softmax(h, dim=-1)
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0))
        cumheights = (top - bottom) * cumheights + bottom
        # rounding to prevent errors
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        widths = self.min_w + (1 - self.min_w * self.num_bins) * F.softmax(w, dim=-1)
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0))
        cumwidths = (right - left) * cumwidths + left
        # rounding to prevent errors
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        delta = heights / widths

        derivs = self.min_d + torch.as_tensor(F.softplus(d))

        # get indices from binary search
        if inverse:
            bin_idx = self.searchsorted(cumheights, data)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, data)[..., None]

        # gather values at last dim, and squeeze last dim
        bin_H = cumheights.gather(-1, bin_idx)[..., 0]
        bin_W = cumwidths.gather(-1, bin_idx)[..., 0]
        bin_h = heights.gather(-1, bin_idx)[..., 0]
        bin_w = widths.gather(-1, bin_idx)[..., 0]
        bin_Δ = delta.gather(-1, bin_idx)[..., 0]
        bin_d = derivs.gather(-1, bin_idx)[..., 0]
        bin_d_plus_one = derivs[..., 1:].gather(-1, bin_idx)[..., 0]

        q = bin_d + bin_d_plus_one - 2 * bin_Δ

        if inverse:
            r = data - bin_H # offset within the bin
            rq = r * q
            a = rq + bin_h * (bin_Δ - bin_d)
            b = bin_h * bin_d - rq
            c = -bin_Δ * r
            discriminant_sqrt = (b.pow(2) - 4 * a * c).relu().sqrt()
            θ = (2 * c) / (-b - discriminant_sqrt)
            θ_one_minus_θ = θ * (1 - θ)

            outputs = bin_W + θ * bin_w

            denominator = bin_Δ + q * θ_one_minus_θ
            derivative_numerator = bin_Δ.pow(2) * (bin_d_plus_one * θ.pow(2) + 2 * bin_Δ * θ_one_minus_θ + bin_d * (1 - θ).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, -logabsdet

        else:
            r = data - bin_W # offset within the bin
            θ = r / bin_w
            θ_one_minus_θ = θ * (1 - θ)

            numerator = bin_h * (bin_Δ * θ.pow(2) + bin_d * θ_one_minus_θ)
            denominator = bin_Δ + q * θ_one_minus_θ
            outputs = bin_H + numerator / denominator

            derivative_numerator = bin_Δ.pow(2) * (bin_d_plus_one * θ.pow(2) + 2 * bin_Δ * θ_one_minus_θ + bin_d * (1 - θ).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, logabsdet

    @staticmethod
    def searchsorted(
        bin_locations: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
