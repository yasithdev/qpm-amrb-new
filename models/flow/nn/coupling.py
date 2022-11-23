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

    @abstractmethod
    def _cforward(
        self,
        xT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    @abstractmethod
    def _cinverse(
        self,
        zT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of the coupling transform."""
        raise NotImplementedError()

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() in [2, 4], "Inputs must be a 2D or a 4D tensor."

        xI = x[:, self.i_channels, ...]
        xT = x[:, self.t_channels, ...]

        params = self.coupling_net(xI, c)
        zI = xI
        zT, logabsdet = self._cforward(xT, params)

        z = torch.empty_like(x)
        z[:, self.i_channels, ...] = zI
        z[:, self.t_channels, ...] = zT

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() in [2, 4], "Inputs must be a 2D or a 4D tensor."

        zI = z[:, self.i_channels, ...]
        zT = z[:, self.t_channels, ...]

        xI = zI
        params = self.coupling_net(xI, c)
        xT, logabsdet = self._cinverse(zT, params)

        x = torch.empty_like(z)
        x[:, self.i_channels, ...] = xI
        x[:, self.t_channels, ...] = xT

        return x, logabsdet


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
            num_layers=num_layers,
            in_channels=nI,
            out_channels=nT * 2,
        )

        super().__init__(
            i_channels=i_channels,
            t_channels=t_channels,
            coupling_net=coupling_net,
        )

        self.nI = nI
        self.nT = nT

    def _cforward(
        self,
        xT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the coupling transform."""

        s, t = torch.chunk(params, 2, dim=1)
        zT = s.exp() * xT + t
        logabsdet = s.sum([1, 2, 3])

        return zT, logabsdet

    def _cinverse(
        self,
        zT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of the coupling transform."""

        s, t = torch.chunk(params, 2, dim=1)
        xT = (zT - t) / s.exp()
        logabsdet = -s.sum([1, 2, 3])

        return xT, logabsdet


class RQSCoupling(CouplingTransform):
    """
    (p1) ---------------(p1)
                ↓
            ([w,h,d])        <- piecewise RQS functions
                ↓
    (p2) ------[x]------(p2)
    """

    def __init__(
        self,
        i_channels: torch.Tensor,
        t_channels: torch.Tensor,
        num_bins: int = 10,
        bound: float = 2.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ) -> None:

        nI = len(i_channels)
        nT = len(t_channels)
        assert nI == nT, f"channel masks have mismatching sizes! ({nI} and {nT})"

        nW = num_bins
        nH = num_bins
        nD = num_bins - 1

        coupling_net = nets.Conv2DResNet(
            num_layers=1,
            in_channels=nI,
            out_channels=nT * (nW + nH + nD),
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
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    @staticmethod
    def searchsorted(
        bin_locations: torch.Tensor,
        inputs: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:

        bin_locations[..., -1] += eps
        return torch.sum(inputs[..., None] >= bin_locations, dim=1) - 1

    def _cforward(
        self,
        xT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self._spline(xT, params, inverse=False)

    def _cinverse(
        self,
        zT: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self._spline(zT, params, inverse=True)

    def _spline(
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
        assert (
            data.size() == params.size()[:-1]
        ), "incompatible shapes for data and params!"

        unbound_h, unbound_w, unbound_d = einops.unpack(
            tensor=params,
            packed_shapes=[
                [self.nH],
                [self.nW],
                [self.nD],
            ],
            pattern="B C H W *",
        )

        outputs, logabsdet = self._unbound_rqs(
            data=data,
            unbound_h=unbound_h,
            unbound_w=unbound_w,
            unbound_d=unbound_d,
            inverse=inverse,
        )
        logabsdet = einops.reduce(logabsdet, "b ... -> b", reduction="sum")

        assert (
            data.size() == outputs.size()
        ), "incompatible shapes for input and output!"

        return outputs, logabsdet

    def _unbound_rqs(
        self,
        data: torch.Tensor,
        unbound_h: torch.Tensor,
        unbound_w: torch.Tensor,
        unbound_d: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        lo = -self.bound
        hi = self.bound
        within_bound = (data >= lo) & (data <= hi)
        out_of_bound = ~within_bound

        outputs = torch.zeros_like(data)
        logabsdet = torch.zeros_like(data)

        # constant-pad the derivative for out-of-bound
        const = np.log(np.exp(1 - self.min_derivative) - 1)
        unbound_d = F.pad(unbound_d, pad=(1, 1), value=const)

        # apply rqs for within_bound data
        if torch.any(within_bound):
            bound_data = data[within_bound]
            bound_h = unbound_h[within_bound, :]
            bound_w = unbound_w[within_bound, :]
            bound_d = unbound_d[within_bound, :]
            kwargs = {
                "data": bound_data,
                "h": bound_h,
                "w": bound_w,
                "d": bound_d,
                "inverse": inverse,
            }
            outputs[within_bound], logabsdet[within_bound] = self._bound_rqs(**kwargs)

        # identity-map out_of_bound data
        if torch.any(out_of_bound):
            outputs[out_of_bound] = data[out_of_bound]

        return outputs, logabsdet

    def _bound_rqs(
        self,
        data: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert w.shape[-1] == self.num_bins

        lo = -self.bound
        hi = self.bound

        within_bound = data.min() >= lo and data.max() <= hi
        assert within_bound, f"Inputs outside domain! [{lo},{hi}]"

        left = bottom = lo
        right = top = hi

        if self.min_bin_width * self.num_bins > 1.0:
            raise ValueError("min_bin_width too large for the number of bins")
        if self.min_bin_height * self.num_bins > 1.0:
            raise ValueError("min_bin_height too large for the number of bins")

        # scale down the variance
        h /= np.sqrt(self.nI)
        w /= np.sqrt(self.nI)

        heights = F.softmax(h, dim=1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        # rounding to prevent errors
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        widths = F.softmax(w, dim=1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        # rounding to prevent errors
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivs = self.min_derivative + F.softplus(d)

        if inverse:
            bin_idx = self.searchsorted(cumheights, data)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, data)[..., None]

        input_cumwidths = cumwidths.gather(1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(1, bin_idx)[..., 0]

        input_derivs = derivs.gather(1, bin_idx)[..., 0]
        input_derivs_plus_one = derivs[..., 1:].gather(1, bin_idx)[..., 0]

        input_heights = heights.gather(1, bin_idx)[..., 0]

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
            outputs = root * input_bin_widths + input_cumwidths

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

            return outputs, -logabsdet

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
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivs_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivs * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, logabsdet
