from typing import List, Tuple

import torch
import torch.distributions

from .transforms import FlowTransform


class SquareNormalizingFlow(FlowTransform):

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        transforms: List[FlowTransform],
    ):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize h=x
        h = x
        # initialize det=1 => log_det=0
        fwd_log_det = torch.zeros(h.size(0), device=x.device)
        # iterate in order
        transform: FlowTransform
        for transform in self.transforms:
            # transform h and accumulate log_det
            h, h_log_det = transform.forward(h)
            fwd_log_det += h_log_det
        z = h
        return z, fwd_log_det

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize h=z
        h = z
        # initialize det=1 => log_det=0
        inv_log_det = torch.zeros(h.size(0), device=z.device)
        # iterate in reverse
        for transform in reversed(self.transforms):
            h, h_log_det = transform.inverse(h)
            inv_log_det += h_log_det
        x = h
        return x, inv_log_det

    # --------------------------------------------------------------------------------------------------------------------------------------------------
