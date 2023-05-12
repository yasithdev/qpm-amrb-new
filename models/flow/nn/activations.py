from typing import Optional, Tuple

import torch
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

        assert x.dim() in [2, 4]

        z = x.tanh()
        logabsdet = torch.log(1 - x.tanh().pow(2)).sum(dim=[1, 2, 3])

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = z.atanh()
        logabsdet = -torch.log(1 - x.tanh().pow(2)).sum(dim=[1, 2, 3])

        return x, logabsdet
