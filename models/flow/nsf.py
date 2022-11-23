"""Implementations of Neural Spline Flows."""

from typing import Optional, Tuple

import torch

from . import Compose, ComposeMultiScale, Flow
from .distributions import Distribution, StandardNormal
from .nn import ConformalConv2D_1x1, RQSCoupling, Squeeze
from .util import decode_mask


class SimpleNSF2D(Flow):

    """
    Same-Scale Neural Spline Flow (2D)

    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_layers: int,
        base_dist: Optional[Distribution] = None,
        include_linear: bool = True,
        num_bins: int = 10,
        bound: int = 2,
    ) -> None:

        transforms = []
        C, H, W = input_shape

        transforms.append(Squeeze(factor=2))
        C, H, W = C * 4, H // 2, W // 2

        mask = torch.zeros(C).bool()
        mask[::2] = True

        for _ in range(num_layers):

            i_channels, t_channels = decode_mask(mask)
            transforms.append(
                RQSCoupling(
                    i_channels=i_channels,
                    t_channels=t_channels,
                    num_bins=num_bins,
                    bound=bound,
                )
            )
            mask = ~mask

            if include_linear:
                transforms.append(ConformalConv2D_1x1(C))

        if base_dist is None:
            base_dist = StandardNormal(C, H, W)

        super().__init__(
            base_dist=base_dist,
            transform=Compose(*transforms),
        )


class MultiScaleNSF2D(Flow):

    """
    Multi-Scale Neural Spline Flow

    """

    def __init__(
        self,
        num_levels: int,
        input_shape: Tuple[int, ...],
        num_layers: int,
        base_dist: Optional[Distribution] = None,
        # common
        include_linear: bool = True,
        num_bins: int = 10,
        bound: float = 2,
    ) -> None:

        assert num_levels > 0, "You need at least one level!"

        transforms = []
        C, H, W = input_shape

        for _ in range(num_levels):

            level_transforms = []

            level_transforms.append(Squeeze(factor=2))
            C, H, W = C * 4, H // 2, W // 2

            mask = torch.zeros(C).bool()
            mask[::2] = True

            for _ in range(num_layers):

                i_channels, t_channels = decode_mask(mask)
                level_transforms.append(
                    RQSCoupling(
                        i_channels=i_channels,
                        t_channels=t_channels,
                        num_bins=num_bins,
                        bound=bound,
                    )
                )
                mask = ~mask

                if include_linear:
                    level_transforms.append(ConformalConv2D_1x1(C))

            transforms.append(Compose(*level_transforms))

        if base_dist is None:
            base_dist = StandardNormal(C, H, W)

        super().__init__(
            base_dist=base_dist,
            transform=ComposeMultiScale(*transforms),
        )
