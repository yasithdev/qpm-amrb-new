from functools import partial
from typing import Callable, Optional
import einops
import math

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
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)])
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList([nn.Linear(features, features) for _ in range(2)])
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
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
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


class Conv2DResBlock(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        context_channels: Optional[int] = None,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        zero_initialization: bool = True,
        reshape: Optional[str] = None,
        needs_adjustment: bool = False,
    ) -> None:
        super().__init__()

        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.reshape = reshape
        self.needs_adjustment = needs_adjustment

        if context_channels is not None:
            self.context_layer = nn.Conv2d(
                in_channels=context_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
            self.bn2 = nn.BatchNorm2d(hidden_channels, eps=1e-3)

        if reshape is None:
            op = partial(nn.Conv2d, stride=1)
        elif reshape == "down": # d_out = (d_in + 1) // 2
            op = partial(nn.Conv2d, stride=2)
        elif reshape == "up":   # d_out = (d_in * 2 - 1)
            op = partial(nn.ConvTranspose2d, stride=2, output_padding=int(self.needs_adjustment))
        else:
            raise ValueError(reshape)
        
        self.convr = op(in_channels, out_channels, kernel_size=1)
        self.conv1 = op(in_channels, hidden_channels, kernel_size=3, padding=1, groups=2)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        if zero_initialization:
            init.uniform_(self.conv2.weight, -1e-3, 1e-3)  # type: ignore
            init.uniform_(self.conv2.bias, -1e-3, 1e-3)  # type: ignore

        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temps = inputs
        if self.use_batch_norm:
            temps = self.bn1(temps)
        temps = self.activation(temps)
        temps = self.conv1(temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        if self.use_batch_norm:
            temps = self.bn2(temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv2(temps)
        output = temps + self.convr(inputs)
        return output


class Conv2DAttention(nn.Module):
    def __init__(
        self,
        in_h: int,
        in_w: int,
        num_heads: int,
        in_features: int,
        out_features: int,
        head_features: int,
    ) -> None:
        super().__init__()
        
        self.in_h = in_h
        self.in_w = in_w
        self.nh = num_heads
        self.di = in_features
        self.do = out_features
        self.dh = head_features

        self.rh = torch.nn.Parameter(torch.randn(self.in_h, self.dh))
        self.rw = torch.nn.Parameter(torch.randn(self.in_w, self.dh))

        self.conv_qkv = torch.nn.Conv2d(self.di, self.nh * self.dh * 3, kernel_size=1)

        self.wp = torch.nn.Parameter(torch.randn(self.nh * self.dh, self.do))
        self.nc = 1 / (self.dh**0.5)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # validation
        assert len(input.shape) == 4
        (h, w) = input.shape[2:]

        # relative position embeddings
        assert self.in_h == h
        assert self.in_w == w
        rel_h = self.rh.tanh().cumsum(dim=0).view(h, 1, self.dh) / (h ** 0.5)
        rel_w = self.rw.tanh().cumsum(dim=0).view(1, w, self.dh) / (w ** 0.5)
        rel = einops.rearrange(rel_h + rel_w, "h w d -> 1 1 (h w) d")

        # attention
        features = self.conv_qkv(input)
        features = einops.rearrange(features, "b (a d) h w -> b a (h w) d", a=self.nh * 3)
        (q, k, v) = torch.chunk(features, 3, dim=1)
        sim = einops.einsum((q + rel), (k + rel), "b a s d, b a S d -> b a s S")
        att = F.softmax(sim * self.nc, dim=-1)
        mha = einops.einsum(att, v, "b a s S, b a S d -> b a s d")
        mha = einops.rearrange(mha, "b a s d -> b s (a d)")
        proj = einops.einsum(mha, self.wp, "b s ad, ad C -> b s C")
        out = einops.rearrange(proj, "b (h w) C -> b C h w", h=h, w=w)
        return out

class Conv2DResNet(nn.Module):
    """"""

    def __init__(
        self,
        in_h: int,
        in_w: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        context_channels: Optional[int] = None,
        activation: Callable = F.tanh,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        use_attention: bool = False,
        num_groups: int = 1,
    ) -> None:
        super().__init__()

        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention

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
        self.down_block =  Conv2DResBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
            out_channels=hidden_channels,
            context_channels=context_channels,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            reshape="down",
            needs_adjustment=in_h % 2 == 0,
        )

        self.up_block = Conv2DResBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
            out_channels=hidden_channels,
            context_channels=context_channels,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            reshape="up",
            needs_adjustment=in_h % 2 == 0,
        )

        if self.use_attention:
            self.attention_layer = Conv2DAttention(
                in_h=in_h,
                in_w=in_w,
                num_heads=4,
                in_features=hidden_channels,
                out_features=hidden_channels,
                head_features=32,
            )
        else:
            self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0, groups=num_groups)

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # initial 1x1conv transform
        if context is None:
            temp1 = self.initial_layer(inputs)
        else:
            temp1 = self.initial_layer(torch.cat((inputs, context), dim=1))
        # spatial bottleneck transform
        temp2 = self.down_block(temp1, context)
        temp3 = self.up_block(temp2, context)
        # unet style skip connection
        temps = temp1 + temp3 
        # spatial attention transform
        if self.use_attention:
            temps_att = self.attention_layer(temps)
            temps = F.glu(torch.concat([temps, temps_att], dim=1), dim=1)
        # final 1x1conv transform
        output = self.final_layer(temps)
        return output
