"""Implementations of Neural Spline Flows."""

from typing import Callable, Optional, Tuple

import torch
from torch.nn import functional as F

from . import Compose, ComposeMultiScale, Flow
from .distributions import Distribution, StandardNormal
from .nets import ConvResNet, ResNet
from .nn import Conv2D_1x1, LULinear, RQSCoupling, Squeeze


class SimpleNSF2D(Flow):

    """
    Same-Scale Neural Spline Flow (2D)

    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_features: int,
        num_layers: int,
        num_blocks: int,
        base_distribution: Optional[Distribution] = None,
        include_linear: bool = True,
        num_bins: int = 11,
        tail_bound: int = 10,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
    ) -> None:

        if base_distribution is None:
            base_distribution = StandardNormal(*input_shape)

        kwargs = {
            "hidden_features": hidden_features,
            "num_blocks": num_blocks,
            "activation": activation,
            "dropout_probability": dropout_probability,
            "use_batch_norm": batch_norm_within_layers,
        }

        transforms = []
        C, H, W = input_shape

        transforms.append(Squeeze(factor=2))
        C, H, W = C * 4, H // 2, W // 2

        mask = torch.ones(input_shape)
        mask[::2] = -1

        for _ in range(num_layers):

            transforms.append(
                RQSCoupling(
                    mask=mask,
                    transform=ResNet(in_features=C, out_features=C, **kwargs),
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    tails="linear",
                )
            )
            mask *= -1

            if include_linear:
                transforms.append(LULinear(C))

        super().__init__(
            base_dist=base_distribution,
            transform=Compose(*transforms),
        )


class MultiScaleNSF2D(Flow):

    """
    Multi-Scale Neural Spline Flow

    """

    def __init__(
        self,
        num_levels: int,
        # common
        input_shape: Tuple[int, ...],
        hidden_features: int,
        num_layers: int,
        num_blocks: int,
        base_distribution: Optional[Distribution] = None,
        include_linear: bool = True,
        num_bins: int = 11,
        tail_bound: float = 10,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
    ) -> None:

        assert num_levels > 0, "You need at least one level!"

        if base_distribution is None:
            base_distribution = StandardNormal(*input_shape)

        kwargs = {
            "hidden_channels": hidden_features,
            "num_blocks": num_blocks,
            "activation": activation,
            "dropout_probability": dropout_probability,
            "use_batch_norm": batch_norm_within_layers,
        }

        transforms = []
        C, H, W = input_shape

        for _ in range(num_levels):

            level_transforms = []

            level_transforms.append(Squeeze(factor=2))
            C, H, W = C * 4, H // 2, W // 2

            mask = torch.ones(C)
            mask[::2] = -1

            for _ in range(num_layers):

                level_transforms.append(
                    RQSCoupling(
                        mask=mask,
                        transform=ConvResNet(in_channels=C, out_channels=C, **kwargs),
                        num_bins=num_bins,
                        tail_bound=tail_bound,
                        tails="linear",
                    )
                )
                mask *= -1

                if include_linear:
                    linear_transform = Conv2D_1x1(C)
                    level_transforms.append(linear_transform)

            transforms.append(Compose(*level_transforms))

        super().__init__(
            base_dist=base_distribution,
            transform=ComposeMultiScale(*transforms),
        )
