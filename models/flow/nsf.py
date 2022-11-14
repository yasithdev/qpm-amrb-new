"""Implementations of Neural Spline Flows."""

from typing import Callable, Optional, Tuple

import torch
from torch.nn import functional as F

from . import Compose, ComposeMultiScale, FlowTransform
from .distributions import Distribution, StandardNormal
from .nets import ConvResidualNet, ResidualNet
from .nn import (
    LULinear,
    OneByOneConvolution,
    PiecewiseRationalQuadraticCouplingTransform,
    RandomPermutation,
    SqueezeTransform,
)


class SimpleNSF(FlowTransform):
    """"""

    def __init__(
        self,
        features: Tuple[int, ...],
        hidden_features: int,
        num_layers: int,
        num_blocks_per_layer: int,
        base_distribution: Optional[Distribution] = None,
        include_linear: bool = True,
        num_bins: int = 11,
        tail_bound: int = 10,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
    ) -> None:

        coupling_constructor = PiecewiseRationalQuadraticCouplingTransform
        mask = torch.ones(features)
        mask[::2] = -1
        if base_distribution is None:
            base_distribution = StandardNormal(*features)

        def create_resnet(
            in_features: int,
            out_features: int,
        ) -> torch.nn.Module:

            return ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            coupling_transform = coupling_constructor(
                mask=mask,
                transform_net_create_fn=create_resnet,
                tails="linear",
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            layers.append(coupling_transform)
            mask *= -1

            if include_linear:
                linear_transform = Compose(
                    RandomPermutation(features=features),
                    LULinear(features, identity_init=True),
                )
                layers.append(linear_transform)

        super().__init__(
            transform=Compose(*layers),
            distribution=base_distribution,
        )


class MultiscaleNSF(FlowTransform):
    """"""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_channels: int,
        num_levels: int,
        num_layers_per_level: int,
        num_blocks_per_layer: int,
        base_distribution: Optional[Distribution] = None,
        include_linear: bool = True,
        num_bins: int = 11,
        tail_bound: float = 1.0,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        presqueeze: bool = False,
        linear_cache: bool = False,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:

        channels, height, width = input_shape
        dimensionality = channels * height * width
        coupling_constructor = PiecewiseRationalQuadraticCouplingTransform

        if base_distribution is None:
            base_distribution = StandardNormal(dimensionality)

        def create_resnet(
            in_channels: int,
            out_channels: int,
        ) -> torch.nn.Module:

            return ConvResidualNet(
                in_channels,
                out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        assert num_levels > 0, "You need at least one level!"

        for level in range(num_levels):
            layers = []

            channels, height, width = input_shape
            if level > 0 or presqueeze:
                squeeze_transform = SqueezeTransform(factor=2)
                layers.append(squeeze_transform)

                channels *= 4
                height //= 2
                width //= 2
                input_shape = channels, height, width

            mask = torch.ones(channels)
            mask[::2] = -1

            for _ in range(num_layers_per_level):
                coupling_transform = coupling_constructor(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                    tails="linear",
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                )
                layers.append(coupling_transform)
                mask *= -1

                if include_linear:
                    linear_transform = OneByOneConvolution(
                        channels, using_cache=linear_cache
                    )
                    layers.append(linear_transform)

            transform = ComposeMultiScale(num_levels, out_shape=out_shape)
            input_shape = transform.add_transform(Compose(*layers), input_shape)

        super().__init__(
            transform=transform,
            distribution=base_distribution,
        )
