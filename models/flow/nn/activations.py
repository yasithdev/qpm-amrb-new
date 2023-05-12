from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from ..base import FlowTransform


class Tanh(FlowTransform):
    def __init__(
        self,
    ):

        super().__init__()

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        assert x.dim() in [2, 4]

        z = x.tanh()
        # logabsdet = torch.log(1 - x.tanh().pow(2)).sum(dim=[1, 2, 3])

        return z, z.new_zeros(B)

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = z.size(0)
        assert z.dim() in [2, 4]

        x = z.atanh()
        # logabsdet = -torch.log(1 - x.tanh().pow(2)).sum(dim=[1, 2, 3])

        return x, x.new_zeros(B)


class LeakyReLU(FlowTransform):
    def __init__(
        self,
    ):
        super().__init__()
        self.neg_slope = 0.01

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        assert x.dim() in [2, 4]

        z = F.leaky_relu(x, negative_slope=self.neg_slope)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)
        assert z.dim() in [2, 4]

        x = F.leaky_relu(z, negative_slope=1 / self.neg_slope)
        logabsdet = z.new_zeros(B)

        return x, logabsdet
