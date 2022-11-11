from typing import Optional, Tuple

import numpy as np
import torch

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

        self.num_channels = num_channels
        w_init = np.random.randn(self.num_channels, self.num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = torch.nn.Parameter(torch.from_numpy(w_init))

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # convolution
        weight = self.weight.view(self.num_channels, self.num_channels, 1, 1)
        z = torch.nn.functional.conv2d(x, weight)

        logabsdet = torch.linalg.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # inverse convolution
        weight = torch.linalg.inv(self.weight).view(
            self.num_channels, self.num_channels, 1, 1
        )
        x = torch.nn.functional.conv2d(z, weight)

        logabsdet = torch.slogdet(self.weight)[1] * z.size(2) * z.size(3)

        return x, -logabsdet
