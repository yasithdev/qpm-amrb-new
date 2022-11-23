from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FlowTransform


class Conv2D_1x1(FlowTransform):

    """
    1x1 convolution with an orthogonally initialized kernel

    (*CHW) <-> [conv] <-> (*CHW)
    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:

        """
        Args:
            num_channels (int): Number of input/output channels (kept same for invertibility)
        """

        super().__init__()

        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.size()

        # convolution
        shape = (C, C, 1, 1)
        weight = self.weight
        z = F.conv2d(x, weight.view(shape))
        logabsdet = x.new_ones(B) * torch.slogdet(weight)[1] * H * W

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = z.size()

        # inverse convolution
        shape = (C, C, 1, 1)
        weight_inv = torch.inverse(self.weight.double()).float()
        x = F.conv2d(z, weight_inv.view(shape))
        logabsdet = z.new_ones(B) * torch.slogdet(weight_inv)[1] * H * W

        return x, logabsdet
