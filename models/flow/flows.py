from typing import Optional, Tuple

import torch

from . import FlowTransform


class Compose(FlowTransform):
    """
    Composition of Flow Transformations

    """

    def __init__(
        self,
        *transforms: FlowTransform,
    ) -> None:

        super().__init__()

        self.transforms = torch.nn.ModuleList(transforms)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # initialize h=x
        h = x

        # initialize det=1 => log_det=0
        sum_logabsdet = torch.zeros(h.size(0), device=x.device)

        # iterate in order
        for transform in self.transforms:
            assert isinstance(transform, FlowTransform)
            # transform h and accumulate log_det
            h, logabsdet = transform.forward(h)
            sum_logabsdet += logabsdet

        z = h
        return z, sum_logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # initialize h=z
        h = z

        # initialize det=1 => log_det=0
        sum_logabsdet = torch.zeros(h.size(0), device=z.device)

        # iterate in reverse
        for transform in reversed(self.transforms):
            assert isinstance(transform, FlowTransform)

            # transform h and accumulate log_det
            h, logabsdet = transform.inverse(h)
            sum_logabsdet += logabsdet
        x = h
        return x, sum_logabsdet
