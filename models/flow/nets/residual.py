from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class ResBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features: int,
        context_features: Optional[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        zero_initialization: bool = True,
    ) -> None:

        super().__init__()

        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)  # type: ignore
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)  # type: ignore

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_blocks: int = 2,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:

        super().__init__()

        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


class ConvResBlock(nn.Module):
    """"""

    def __init__(
        self,
        channels: int,
        context_channels: Optional[int] = None,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        zero_initialization: bool = True,
    ) -> None:

        super().__init__()

        self.activation = activation

        if context_channels is not None:
            self.context_layer = nn.Conv2d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm2d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)  # type: ignore
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)  # type: ignore

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ConvResNet(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        context_channels: Optional[int] = None,
        num_blocks: int = 2,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:

        super().__init__()

        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                ConvResBlock(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1, padding=0
        )

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class Conv2DResNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        """

        :param num_layers: number of conv2d layers
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel height and width
        """
        super().__init__()

        assert num_layers >= 1

        self.num_layers = num_layers

        self.w = nn.ParameterList()
        self.b = nn.ParameterList()

        kH = kW = kernel_size
        c_in = in_channels
        c_out = out_channels

        self.w.append(nn.Parameter(torch.randn(c_out, c_in, kH, kW)))
        self.b.append(nn.Parameter(torch.randn(c_out)))

        for _ in range(1, self.num_layers):
            self.w.append(nn.Parameter(torch.randn(c_out, c_out, kH, kW)))
            self.b.append(nn.Parameter(torch.randn(c_out)))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        st = x
        for i in range(self.num_layers):
            w, b = self.w[i], self.b[i]
            st_res = nn.functional.conv2d(
                input=st,
                weight=w,
                bias=b,
                padding="same",
            )
            if i == 0:
                st = torch.tanh(st_res)
            else:
                st = torch.tanh(st_res + st)

        return st
